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


def _forward(model, payload, device):
    if "selector_hidden" in payload:
        return payload["selector_hidden"].to(device)
    noise = payload["noise_embedding"].to(device)
    context = payload["context"].to(device)
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
    trainable = [projection]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * (len(dataset) - len(val_ids)))
    )
    checkpoint_path = output / "checkpoint.pt"
    start_epoch = 0
    step = 0
    best = math.inf
    best_projection = projection.detach().cpu().clone()
    history = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        projection.data.copy_(checkpoint["projection"].to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        step = int(checkpoint["step"])
        best = float(checkpoint["best"])
        best_projection = checkpoint["best_projection"].cpu()
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
            best_projection = projection.detach().cpu().clone()
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "projection": projection.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1,
                "step": step,
                "best": best,
                "best_projection": best_projection,
                "history": history,
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
    projection.data.copy_(best_projection.to(device))
    save_file(
        {"weight": projection.detach().cpu().contiguous()},
        str(output / "distilled_hidden_projection.safetensors"),
    )
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
        "loss_reduction": args.loss_reduction,
        "history": history,
        "objective": (
            "Teacher-greedy cross-entropy over target top-k unary-plus-edge scores"
        ),
        "non_claims": [
            "Validation is reconstruction loss on captured cycles, not live acceptance.",
            "The transformer, target projection, and selector codebooks are frozen.",
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
