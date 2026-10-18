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


class CycleDataset(torch.utils.data.Dataset[tuple[dict[str, torch.Tensor], int]]):
    def __init__(self, root: Path, limit: int = 0):
        self.paths = _cycle_paths(root)
        if limit:
            self.paths = self.paths[:limit]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int]:
        data = load_file(str(self.paths[index]))
        prompt_index = 0
        return data, prompt_index


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


def _selector_loss(model, payload, hidden, device, reduction="mean"):
    top_k = int(model.candidate_selector.top_k)
    candidate_ids = payload["target_top_ids"][1:].to(device).long()[None, :, :top_k]
    unary = payload["target_top_values"][1:].to(device).float()[None, :, :top_k]
    target_ids = payload["target_greedy"][1:].to(device).long()[None]
    projected = model.candidate_selector.hidden_projection(hidden)
    successor = model.candidate_selector.successor_codebook(candidate_ids)
    predecessor = payload["anchor"].to(device).long()
    losses = []
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
        losses.append(
            F.cross_entropy(
                scores,
                target_index,
            )
        )
        predecessor = target_ids[:, position]
    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses.sum()


def run(args):
    device = args.device
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
    dataset = CycleDataset(Path(args.capture_dir), args.limit)
    if not dataset:
        raise RuntimeError("empty capture")
    ids = list(range(len(dataset)))
    random.Random(args.seed).shuffle(ids)
    split = max(1, int(len(ids) * args.validation_fraction))
    val_ids = set(ids[:split])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainable = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * (len(dataset) - len(val_ids)))
    )
    checkpoint_path = output / "checkpoint.pt"
    start_epoch = 0
    step = 0
    best = math.inf
    best_parameters = _parameter_snapshot(model)
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
        history = list(checkpoint["history"])
        print({"resumed_epoch": start_epoch, "best_validation": best}, flush=True)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        order = [i for i in ids if i not in val_ids]
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            payload, _ = dataset[index]
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                hidden = _forward(model, payload, device)
                loss = _selector_loss(
                    model, payload, hidden, device, args.loss_reduction
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_every == 0:
                history.append({"step": step, "loss": float(loss.detach())})
                print(json.dumps(history[-1]), flush=True)

        model.eval()
        total = 0.0
        count = 0
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16),
        ):
            for index in sorted(val_ids):
                payload, _ = dataset[index]
                hidden = _forward(model, payload, device)
                loss = _selector_loss(
                    model, payload, hidden, device, args.loss_reduction
                )
                total += float(loss)
                count += 1
        validation = total / max(1, count)
        print({"epoch": epoch, "validation": validation}, flush=True)
        history.append({"epoch": epoch, "validation": validation})
        if validation < best:
            best = validation
            best_parameters = _parameter_snapshot(model)
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
    save_file(model.state_dict(), str(output / "model.safetensors"))
    receipt = {
        "schema_version": 1,
        "cycles": len(dataset),
        "validation_cycles": len(val_ids),
        "epochs": args.epochs,
        "best_validation": best,
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
        "objective": (
            "Teacher-greedy cross-entropy over target top-k unary-plus-edge scores"
        ),
        "non_claims": [
            "Validation is reconstruction loss on captured cycles, not live acceptance.",
            "Selector codebooks and modules without an adapter remain frozen.",
            "The saved state is a PyTorch training checkpoint, not an MLX artifact.",
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
