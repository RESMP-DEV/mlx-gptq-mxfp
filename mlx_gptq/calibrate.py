"""Stage A CLI: GPTQ calibration on CUDA (or CPU for smoke tests).

Example (on the CUDA box):

    python -m mlx_gptq.calibrate \
        --model /models/Qwen3-30B-A3B --output ./calib-qwen3-30b \
        --dataset /data/calibration_datav3.txt \
        --nsamples 256 --seqlen 2048 --bits 4 --group-size 64 \
        --devices cuda:0,cuda:1,cuda:2,cuda:3
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil

import numpy as np
import torch

log = logging.getLogger("mlx_gptq")


def build_model(model_path: str, dtype: str, trust_remote_code: bool):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config._experts_implementation = "eager"  # force the F.linear expert loop
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    model.config.use_cache = False
    return model


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def parse_overrides(specs):
    out = []
    for spec in specs or []:
        pattern, _, bits = spec.rpartition("=")
        out.append((re.compile(pattern), int(bits)))
    return out


def parse_vram_budgets(spec: str, devices: list[str]) -> dict[str, float]:
    values = [float(value.strip()) for value in spec.split(",")]
    if len(values) == 1:
        values *= len(devices)
    if len(values) != len(devices):
        raise ValueError(
            "--vram-gb must be one value or one comma-separated value per device"
        )
    if any(value <= 0 for value in values):
        raise ValueError("--vram-gb values must be positive")
    return dict(zip(devices, values))


def main(argv=None):
    from .artifacts import ArtifactReader, ArtifactWriter
    from .data import load_calibration
    from .grid import SUPPORTED_BITS, SUPPORTED_MODES, mode_defaults
    from .sequential import DEFAULT_EXCLUDE, Pipeline, RawShardIndex

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model id or local dir")
    ap.add_argument("--output", required=True, help="artifact output dir")
    ap.add_argument(
        "--dataset",
        default=None,
        help="comma-separated: file.txt | file.jsonl[:field] | hf:name[:split[:field]]",
    )
    ap.add_argument(
        "--calibration-tokens",
        default=None,
        help="pre-tokenized int64 .npy windows; copied into the artifact receipt",
    )
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4, help="sequences per forward")
    ap.add_argument("--mode", default="affine", choices=SUPPORTED_MODES)
    ap.add_argument(
        "--mxfp-algorithm",
        default="auto",
        choices=("auto", "groupwise", "safe-scale-search"),
        help="MXFP GPTQ variant; auto uses safe scale search for MXFP4 and native groupwise GPTQ for MXFP8",
    )
    ap.add_argument("--bits", type=int, default=None, choices=SUPPORTED_BITS)
    ap.add_argument("--group-size", type=int, default=None, choices=(32, 64, 128))
    ap.add_argument("--clip", default="mse", choices=("mse", "minmax"))
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument(
        "--devices",
        default=None,
        help="comma-separated, first is the forward device (default: all CUDA, else cpu)",
    )
    ap.add_argument(
        "--vram-gb",
        default="16",
        help=(
            "expert-solve budget in GiB: one value for every device or a "
            "comma-separated value corresponding to --devices"
        ),
    )
    ap.add_argument("--layers-attr", default="model.layers")
    ap.add_argument(
        "--checkpoint-model-prefix",
        default=None,
        help=(
            "raw checkpoint prefix corresponding to the runtime model root; "
            "e.g. model.language_model when AutoModelForCausalLM exposes model"
        ),
    )
    ap.add_argument(
        "--artifact-layers-prefix",
        default=None,
        help=(
            "layer prefix expected by the packed MLX checkpoint; defaults to "
            "--layers-attr"
        ),
    )
    ap.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help="regex of layer-relative module names to leave unquantized",
    )
    ap.add_argument(
        "--override",
        action="append",
        metavar="REGEX=BITS",
        help="per-tensor bit override, e.g. '.*down_proj.*=8' (repeatable)",
    )
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    default_group, default_bits = mode_defaults(args.mode)
    args.group_size = args.group_size or default_group
    args.bits = args.bits or default_bits
    if args.mode in ("mxfp4", "mxfp8"):
        if (args.group_size, args.bits) != (default_group, default_bits):
            ap.error(
                f"{args.mode} requires --group-size {default_group} "
                f"and --bits {default_bits}"
            )
        if args.override:
            ap.error("bit overrides are not supported for native MXFP modes")
        if args.mxfp_algorithm == "auto":
            args.mxfp_algorithm = (
                "safe-scale-search" if args.mode == "mxfp4" else "groupwise"
            )
    else:
        args.mxfp_algorithm = "none"
    if args.dataset is None and args.calibration_tokens is None:
        ap.error("one of --dataset or --calibration-tokens is required")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.devices:
        args.devices = [d.strip() for d in args.devices.split(",")]
    elif torch.cuda.is_available():
        args.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        args.devices = ["cpu"]
    try:
        args.vram_gb = parse_vram_budgets(args.vram_gb, args.devices)
    except ValueError as exc:
        ap.error(str(exc))
    args.overrides = parse_overrides(args.override)
    log.info("devices: %s (forward on %s)", args.devices, args.devices[0])
    log.info("expert-solve VRAM budgets: %s", args.vram_gb)

    model_dir = resolve_model_dir(args.model)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=args.trust_remote_code
    )

    tokens_path = os.path.join(args.output, "calib_tokens.npy")
    if os.path.exists(tokens_path):
        samples = torch.from_numpy(np.load(tokens_path))
        log.info("resuming with existing calib_tokens.npy: %s", tuple(samples.shape))
    elif args.calibration_tokens is not None:
        source_tokens = os.path.abspath(args.calibration_tokens)
        samples_np = np.load(source_tokens)
        if samples_np.ndim != 2:
            raise ValueError(
                f"calibration tokens must be rank 2, got shape {samples_np.shape}"
            )
        os.makedirs(args.output, exist_ok=True)
        shutil.copyfile(source_tokens, tokens_path)
        samples = torch.from_numpy(samples_np)
        log.info("using pre-tokenized calibration windows: %s", tuple(samples.shape))
    else:
        samples = load_calibration(
            tokenizer, args.dataset, args.nsamples, args.seqlen, args.seed
        )
        os.makedirs(args.output, exist_ok=True)
        np.save(tokens_path, samples.numpy())

    if samples.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"calibration tokens must be integer IDs, got {samples.dtype}")
    runtime_vocab_size = len(tokenizer)
    if int(samples.min()) < 0 or int(samples.max()) >= runtime_vocab_size:
        raise ValueError(
            f"token IDs [{int(samples.min())}, {int(samples.max())}] exceed "
            f"runtime tokenizer vocabulary {runtime_vocab_size}"
        )
    with open(tokens_path, "rb") as f:
        tokens_sha256 = hashlib.file_digest(f, "sha256").hexdigest()

    model = build_model(model_dir, args.dtype, args.trust_remote_code)
    shards = RawShardIndex(model_dir)

    writer = ArtifactWriter(
        args.output,
        {
            "model": args.model,
            "algorithm": args.mxfp_algorithm if args.mode != "affine" else "gptq",
            "mode": args.mode,
            "bits": args.bits,
            "group_size": args.group_size,
            "storage_dtype": args.dtype,
            "clip": args.clip,
            "damp": args.damp,
            "nsamples": int(samples.shape[0]),
            "seqlen": int(samples.shape[1]),
            "dataset": args.dataset,
            "calibration_tokens": args.calibration_tokens,
            "calibration_tokens_sha256": tokens_sha256,
            "layers_attr": args.layers_attr,
            "checkpoint_model_prefix": args.checkpoint_model_prefix,
            "artifact_layers_prefix": args.artifact_layers_prefix,
        },
    )
    reader = ArtifactReader(args.output) if writer.manifest["layers_done"] else None

    pipe = Pipeline(model, shards, args)
    with torch.no_grad():
        pipe.run(samples, writer, reader_for_resume=reader)
    log.info("artifacts written to %s", args.output)


if __name__ == "__main__":
    main()
