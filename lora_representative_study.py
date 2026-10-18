#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import optuna


def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    rank = trial.suggest_categorical("rank", (4, 8, 16))
    alpha_ratio = trial.suggest_categorical("alpha_ratio", (1.0, 2.0))
    predecessor_mode = trial.suggest_categorical(
        "predecessor_mode", ("teacher", "predicted")
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 2e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.03, log=True)

    output = args.study_dir / f"trial-{trial.number:02d}"
    if output.exists():
        raise RuntimeError(f"trial output already exists: {output}")
    command = [
        str(args.python),
        "-m",
        "mlx_gptq.dflash2_distill",
        "--model",
        str(args.root / "model"),
        "--capture-dir",
        str(args.root / "capture"),
        "--output",
        str(output),
        "--dflash-package",
        str(args.root / "dflash-pytorch"),
        "--initial-projection",
        str(args.root / "initial_selector_projection.safetensors"),
        "--mode",
        "lora",
        "--lora-rank",
        str(rank),
        "--lora-alpha",
        str(rank * alpha_ratio),
        "--device",
        "cuda:0",
        "--epochs",
        "1",
        "--limit",
        str(args.cycles),
        "--stratified-limit",
        "--validation-fraction",
        "0.2",
        "--selection-metric",
        "prefix",
        "--predecessor-mode",
        predecessor_mode,
        "--lr",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--seed",
        str(args.seed),
        "--log-every",
        "128",
        "--no-resume",
    ]
    env = os.environ.copy()
    env.update(
        {
            # Keep the physical GPU in CUDA_VISIBLE_DEVICES and use ordinal zero.
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log_path = args.study_dir / f"trial-{trial.number:02d}.log"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=args.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        trial.set_user_attr("failed_exit_code", completed.returncode)
        trial.set_user_attr("failed_log", str(log_path))
        raise optuna.TrialPruned(
            f"trial command exited {completed.returncode}; see {log_path}"
        )
    receipt = json.loads((output / "receipt.json").read_text())
    result = float(receipt["best_validation"])
    trial.set_user_attr("validation_metrics", receipt["validation_metrics"])
    trial.set_user_attr("initial_metrics", receipt["initial_metrics"])
    trial.set_user_attr("validation_prompts", receipt["validation_prompts"])
    trial.set_user_attr("prompt_count", receipt["prompt_count"])
    shutil.rmtree(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=640)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    args.study_dir.mkdir(parents=True, exist_ok=True)
    args.python = args.root / ".venv/bin/python"
    study = optuna.create_study(
        study_name=f"dflash2-representative-lora-{args.gpu}",
        storage=f"sqlite:///{args.study_dir / 'study.db'}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=4),
        load_if_exists=True,
    )
    queued = (
        {
            "rank": 4,
            "alpha_ratio": 1.0,
            "predecessor_mode": "teacher",
            "learning_rate": 5e-6,
            "weight_decay": 0.01,
        },
        {
            "rank": 4,
            "alpha_ratio": 2.0,
            "predecessor_mode": "predicted",
            "learning_rate": 5e-6,
            "weight_decay": 0.01,
        },
        {
            "rank": 8,
            "alpha_ratio": 1.0,
            "predecessor_mode": "teacher",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
        },
        {
            "rank": 8,
            "alpha_ratio": 2.0,
            "predecessor_mode": "predicted",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
        },
        {
            "rank": 16,
            "alpha_ratio": 1.0,
            "predecessor_mode": "teacher",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
        },
        {
            "rank": 16,
            "alpha_ratio": 2.0,
            "predecessor_mode": "predicted",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
        },
    )
    for params in queued:
        study.enqueue_trial(params, skip_if_exists=True)
    study.optimize(
        lambda trial: run_trial(trial, args),
        n_trials=args.trials,
        gc_after_trial=True,
        catch=(RuntimeError, optuna.TrialPruned),
    )
    trials = sorted(
        (
            {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ),
        key=lambda item: float("inf") if item["value"] is None else item["value"],
    )
    summary = {
        "schema_version": 2,
        "objective": "validation prefix error, prompt-grouped and prompt-stratified",
        "cycles_per_trial": args.cycles,
        "validation_fraction": 0.2,
        "seed": args.seed,
        "best_trial": trials[0],
        "trials": trials,
        "non_claims": [
            "Offline prefix accuracy does not replace parity or live acceptance.",
            "The sweep uses the MXFP4-target capture, not pinned BF16 capture.",
        ],
    }
    (args.study_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
