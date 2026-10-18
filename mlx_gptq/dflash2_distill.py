"""Target-aware fine-tuning for a DFlash2 draft on captured cycles."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .dflash2_calibrate import _cycle_paths, _load_reference


def _stratified_indices(prompt_ids: list[int], limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= len(prompt_ids):
        return list(range(len(prompt_ids)))
    groups: dict[int, list[int]] = {}
    for index, prompt_id in enumerate(prompt_ids):
        groups.setdefault(prompt_id, []).append(index)
    quotas = {
        prompt_id: max(1, int(limit * len(indices) / len(prompt_ids)))
        for prompt_id, indices in groups.items()
    }
    remaining = limit - sum(quotas.values())
    if remaining < 0:
        active = sorted(groups, key=lambda item: quotas[item] / len(groups[item]))
        cursor = 0
        while remaining < 0 and cursor < len(active):
            prompt_id = active[cursor]
            if quotas[prompt_id] > 1:
                quotas[prompt_id] -= 1
                remaining += 1
            cursor += 1
        if remaining < 0:
            raise ValueError("stratified quota could not satisfy the cycle limit")
    fractions = sorted(
        groups,
        key=lambda item: (
            -(limit * len(groups[item]) / len(prompt_ids) - quotas[item]),
            item,
        ),
    )
    for prompt_id in fractions[:remaining]:
        quotas[prompt_id] += 1

    rng = random.Random(seed)
    selected: set[int] = set()
    for prompt_id, indices in groups.items():
        selected.update(rng.sample(indices, quotas[prompt_id]))
    return sorted(selected)


class CycleDataset(torch.utils.data.Dataset[tuple[dict[str, torch.Tensor], int]]):
    def __init__(
        self, root: Path, limit: int = 0, stratified: bool = False, seed: int = 0
    ):
        self.paths = _cycle_paths(root)
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        self.prompt_ids = [int(record["prompt_index"]) for record in index["cycles"]]
        if len(self.prompt_ids) != len(self.paths):
            raise ValueError("capture cycle and prompt-index counts do not match")
        if limit:
            selected = (
                _stratified_indices(self.prompt_ids, limit, seed)
                if stratified
                else list(range(min(limit, len(self.paths))))
            )
            self.paths = [self.paths[index] for index in selected]
            self.prompt_ids = [self.prompt_ids[index] for index in selected]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int]:
        data = load_file(str(self.paths[index]))
        return data, self.prompt_ids[index]


def _prompt_split(
    prompt_ids: list[int], fraction: float, seed: int
) -> tuple[list[int], list[int], set[int]]:
    unique = sorted(set(prompt_ids))
    random.Random(seed).shuffle(unique)
    count = max(1, int(len(unique) * fraction))
    if count >= len(unique):
        raise ValueError("capture has too few prompts for a held-out split")
    validation_prompts = set(unique[:count])
    train = [
        index
        for index, prompt in enumerate(prompt_ids)
        if prompt not in validation_prompts
    ]
    validation = [
        index for index, prompt in enumerate(prompt_ids) if prompt in validation_prompts
    ]
    if not train or not validation:
        raise ValueError("prompt split produced an empty side")
    return train, validation, validation_prompts


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if base.bias is not None:
            raise ValueError("DFlash2 LoRA requires a bias-free linear layer")
        self.base = base
        self.rank = int(rank)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = float(alpha) / float(rank)
        self.a = torch.nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.b = torch.nn.Parameter(
            torch.empty(base.out_features, rank, device=base.weight.device)
        )
        torch.nn.init.kaiming_uniform_(self.a, a=5**0.5)
        torch.nn.init.zeros_(self.b)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        delta = torch.nn.functional.linear(input, self.b @ self.a)
        return self.base(input) + delta * self.scaling


def _forward(model, payload, device):
    if "selector_hidden" in payload:
        return payload["selector_hidden"].to(device)
    noise = payload["noise_embedding"].to(device)
    raw_context = payload.get("raw_context")
    projected_context = payload["context"].to(device)
    context = (
        model.hidden_norm(model.fc(raw_context.to(device)))
        if raw_context is not None
        and raw_context.shape[1] == projected_context.shape[1]
        and raw_context.shape[-1] == model.fc.in_features
        else projected_context
    )
    position_ids = torch.arange(
        context.shape[1] + noise.shape[1], device=device
    ).unsqueeze(0)
    position_embeddings = model.rotary_emb(noise, position_ids)
    hidden = noise
    for layer in model.layers:
        hidden = layer(
            hidden_states=hidden,
            target_hidden=context,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
    return model.norm(hidden)[:, 1:, :]


def _install_lora(model: torch.nn.Module, rank: int, alpha: float) -> tuple[int, int]:
    pending: list[tuple[str, torch.nn.Module]] = [("", model)]
    installed = 0
    trainable = 0
    visited: set[int] = set()
    while pending:
        prefix, parent = pending.pop()
        if id(parent) in visited:
            continue
        visited.add(id(parent))
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if path.endswith("candidate_selector.hidden_projection"):
                continue
            if not isinstance(child, torch.nn.Linear):
                pending.append((f"{path}.", child))
                continue
            adapter = LoRALinear(child, rank, alpha)
            setattr(parent, name, adapter)
            installed += 1
            trainable += adapter.a.numel() + adapter.b.numel()
    return installed, trainable


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_parameters(model: torch.nn.Module, best: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in best.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _lora_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            result[f"{name}.a"] = module.a.detach().cpu().contiguous()
            result[f"{name}.b"] = module.b.detach().cpu().contiguous()
            result[f"{name}.scaling"] = torch.tensor([module.scaling])
    return result


def _merge_lora(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for module in [
            item for item in model.modules() if isinstance(item, LoRALinear)
        ]:
            module.base.weight.add_(module.b @ module.a * module.scaling)


def _remove_lora(model: torch.nn.Module) -> None:
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            setattr(model, name, child.base)
        else:
            _remove_lora(child)


def _selector_loss(
    model,
    payload,
    hidden,
    device,
    reduction="mean",
    predecessor_mode="teacher",
    return_metrics=False,
):
    top_k = int(model.candidate_selector.top_k)
    candidate_ids = payload["target_top_ids"][1:].to(device).long()[None, :, :top_k]
    unary = payload["target_top_values"][1:].to(device).float()[None, :, :top_k]
    target_ids = payload["target_greedy"][1:].to(device).long()[None]
    projected = model.candidate_selector.hidden_projection(hidden)
    successor = model.candidate_selector.successor_codebook(candidate_ids)
    predecessor = payload["anchor"].to(device).long()
    losses = []
    predictions = []
    for position in range(candidate_ids.shape[1]):
        edge = torch.einsum(
            "br,bkr->bk",
            model.candidate_selector.predecessor_codebook(predecessor)
            * projected[:, position],
            successor[:, position],
        )
        matches = candidate_ids[:, position] == target_ids[:, position]
        if not bool(matches.any(dim=1).item()):
            raise RuntimeError("teacher greedy token is absent from its selector set")
        target_index = matches.to(torch.int8).argmax(dim=1)
        scores = edge.float() + unary[:, position]
        prediction_index = scores.argmax(dim=1)
        predictions.append(
            candidate_ids[:, position].gather(1, prediction_index[:, None])
        )
        losses.append(
            F.cross_entropy(
                scores,
                target_index,
            )
        )
        if predecessor_mode == "teacher":
            predecessor = target_ids[:, position]
        elif predecessor_mode == "predicted":
            predecessor = predictions[-1][:, 0].detach()
        else:
            raise ValueError(f"unknown predecessor mode: {predecessor_mode}")
    losses = torch.stack(losses)
    loss = losses.mean() if reduction == "mean" else losses.sum()
    if not return_metrics:
        return loss
    predicted_ids = torch.cat(predictions, dim=1)
    correct = predicted_ids == target_ids
    mismatch = (~correct).long().argmax(dim=1)
    full_match = correct.all(dim=1)
    prefix_lengths = torch.where(full_match, correct.shape[1], mismatch)
    metrics = {
        "loss": loss.detach(),
        "top1_accuracy": correct.flatten().float().mean(),
        "exact_match": full_match.float().mean(),
        "prefix_accuracy": prefix_lengths.float() / correct.shape[1],
    }
    return loss, metrics


def _evaluate_cycles(
    dataset: CycleDataset,
    indices: list[int],
    model: torch.nn.Module,
    device: str,
    *,
    reduction: str,
    predecessor_mode: str,
) -> dict[str, float | dict[str, float]]:
    totals = {
        "loss": 0.0,
        "top1_accuracy": 0.0,
        "exact_match": 0.0,
        "prefix_accuracy": 0.0,
    }
    count = 0
    by_prompt: dict[int, dict[str, float]] = {}
    prompt_counts: dict[int, int] = {}
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16),
    ):
        for index in sorted(indices):
            payload, prompt_id = dataset[index]
            hidden = _forward(model, payload, device)
            _loss, metrics = _selector_loss(
                model,
                payload,
                hidden,
                device,
                reduction=reduction,
                predecessor_mode=predecessor_mode,
                return_metrics=True,
            )
            values = {name: float(tensor) for name, tensor in metrics.items()}
            for name, value in values.items():
                totals[name] += value
            by_prompt.setdefault(prompt_id, {name: 0.0 for name in values})
            prompt_counts.setdefault(prompt_id, 0)
            for name, value in values.items():
                by_prompt[prompt_id][name] += value
            prompt_counts[prompt_id] += 1
            count += 1
    if not count:
        raise RuntimeError("validation set is empty")
    result: dict[str, float | dict[str, float]] = {
        name: total / count for name, total in totals.items()
    }
    result["by_prompt"] = {
        str(prompt_id): {
            name: total / prompt_counts[prompt_id] for name, total in values.items()
        }
        for prompt_id, values in by_prompt.items()
    }
    return result


def _flat_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            result.update(_flat_metrics(f"{prefix}/{name}", value))
        else:
            result[f"{prefix}/{name}"] = float(value)
    return result


def _start_wandb(args: argparse.Namespace, output: Path):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "--wandb-mode is enabled but the wandb package is unavailable"
        ) from error
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        dir=str(output),
        tags=[item for item in args.wandb_tags.split(",") if item],
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )


def run(args):
    device = args.device
    torch.manual_seed(args.seed)
    model = _load_reference(args.model, args.dflash_package, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.mode == "lora":
        adapter_modules, adapter_parameters = _install_lora(
            model, args.lora_rank, args.lora_alpha
        )
    else:
        adapter_modules = 0
        adapter_parameters = 0

    selector = model.candidate_selector.hidden_projection
    selector.weight = torch.nn.Parameter(
        selector.weight.detach().float(), requires_grad=True
    )
    projection = selector.weight
    if args.initial_projection:
        initial = load_file(args.initial_projection)["weight"]
        if tuple(initial.shape) != tuple(projection.shape):
            raise ValueError(
                f"initial projection shape {tuple(initial.shape)} does not match "
                f"{tuple(projection.shape)}"
            )
        projection.data.copy_(initial.to(device).float())
    dataset = CycleDataset(
        Path(args.capture_dir), args.limit, args.stratified_limit, args.seed
    )
    if not dataset:
        raise RuntimeError("empty capture")
    ids, validation_ids, validation_prompts = _prompt_split(
        dataset.prompt_ids, args.validation_fraction, args.seed
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb(args, output)
    trainable = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(ids))
    )
    checkpoint_path = output / "checkpoint.pt"
    start_epoch = 0
    step = 0
    best = math.inf
    best_parameters = _parameter_snapshot(model)
    best_metrics = None
    history = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        projection.data.copy_(checkpoint["projection"].to(device))
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in checkpoint["current_parameters"].items():
                parameters[name].copy_(value.to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        step = int(checkpoint["step"])
        best = float(checkpoint["best"])
        best_parameters = checkpoint["best_parameters"]
        best_metrics = checkpoint["best_metrics"]
        history = list(checkpoint["history"])
        print({"resumed_epoch": start_epoch, "best_validation": best}, flush=True)
    started = time.time()
    initial_metrics = _evaluate_cycles(
        dataset,
        validation_ids,
        model,
        device,
        reduction=args.loss_reduction,
        predecessor_mode=args.predecessor_mode,
    )
    if start_epoch == 0:
        best = (
            float(initial_metrics["loss"])
            if args.selection_metric == "ce"
            else 1.0 - float(initial_metrics["prefix_accuracy"])
        )
        best_parameters = _parameter_snapshot(model)
        best_metrics = initial_metrics
        history.append({"epoch": -1, "validation": best})
        if wandb_run is not None:
            wandb_run.log({"epoch": -1, **_flat_metrics("validation", initial_metrics)})
    for epoch in range(start_epoch, args.epochs):
        order = list(ids)
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            payload, _ = dataset[index]
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                hidden = _forward(model, payload, device)
                loss = _selector_loss(
                    model,
                    payload,
                    hidden,
                    device,
                    args.loss_reduction,
                    args.predecessor_mode,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if wandb_run is not None and step % args.wandb_log_every == 0:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train/loss": float(loss.detach()),
                        "train/grad_norm": float(grad_norm.detach()),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
            if step % args.log_every == 0:
                history.append({"step": step, "loss": float(loss.detach())})
                print(json.dumps(history[-1]), flush=True)

        validation_metrics = _evaluate_cycles(
            dataset,
            validation_ids,
            model,
            device,
            reduction=args.loss_reduction,
            predecessor_mode=args.predecessor_mode,
        )
        validation = (
            float(validation_metrics["loss"])
            if args.selection_metric == "ce"
            else 1.0 - float(validation_metrics["prefix_accuracy"])
        )
        print(
            {"epoch": epoch, "validation": validation, **validation_metrics}, flush=True
        )
        history.append({"epoch": epoch, "validation": validation})
        if wandb_run is not None:
            validation_flat = _flat_metrics("validation", validation_metrics)
            wandb_run.log({"epoch": epoch, "step": step, **validation_flat})
        if validation < best:
            best = validation
            best_parameters = _parameter_snapshot(model)
            best_metrics = validation_metrics
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "projection": projection.detach().cpu(),
                "current_parameters": _parameter_snapshot(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1,
                "step": step,
                "best": best,
                "best_parameters": best_parameters,
                "best_metrics": best_metrics,
                "history": history,
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
    _restore_parameters(model, best_parameters)
    save_file(
        {"weight": selector.weight.detach().cpu().contiguous()},
        str(output / "distilled_hidden_projection.safetensors"),
    )
    if args.mode == "lora":
        save_file(
            _lora_parameters(model),
            str(output / "lora_adapters.safetensors"),
        )
        _merge_lora(model)
        _remove_lora(model)
    save_file(model.state_dict(), str(output / "model.safetensors"))
    receipt = {
        "schema_version": 1,
        "cycles": len(dataset),
        "prompt_count": len(set(dataset.prompt_ids)),
        "stratified_limit": args.stratified_limit,
        "validation_cycles": len(validation_ids),
        "validation_prompts": sorted(validation_prompts),
        "epochs": args.epochs,
        "best_validation": best,
        "best_validation_metrics": best_metrics,
        "selection_metric": args.selection_metric,
        "validation_metrics": validation_metrics,
        "predecessor_mode": args.predecessor_mode,
        "steps": step,
        "elapsed_seconds": time.time() - started,
        "device": device,
        "initial_projection": args.initial_projection,
        "mode": args.mode,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_modules": adapter_modules,
        "trainable_parameters": adapter_parameters + selector.weight.numel(),
        "loss_reduction": args.loss_reduction,
        "history": history,
        "initial_metrics": initial_metrics,
        "wandb": (
            None
            if wandb_run is None
            else {
                "mode": args.wandb_mode,
                "project": args.wandb_project,
                "name": wandb_run.name,
                "id": wandb_run.id,
                "dir": str(wandb_run.dir),
            }
        ),
        "objective": (
            "Teacher-greedy cross-entropy over target top-k unary-plus-edge scores"
        ),
        "non_claims": [
            "Validation metrics are captured-cycle proxies, not live acceptance.",
            "Selector codebooks and modules without an adapter remain frozen.",
            "The saved state is a PyTorch training checkpoint, not an MLX artifact.",
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if wandb_run is not None:
        wandb_run.log({"final/elapsed_seconds": time.time() - started})
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dflash-package", default="/home/kearm/dflash-pytorch")
    parser.add_argument("--initial-projection")
    parser.add_argument("--mode", choices=("selector", "lora"), default="selector")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stratified-limit", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric", choices=("ce", "prefix"), default="prefix"
    )
    parser.add_argument(
        "--predecessor-mode", choices=("teacher", "predicted"), default="teacher"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="dflash2-distillation")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
