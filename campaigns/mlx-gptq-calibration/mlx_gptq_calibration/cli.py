from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .campaign import Campaign, CampaignError, load_campaign, sha256_file
from .hybrid import apply_overlay, fetch_overlay

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "campaigns" / "qwen3.8-27b-mxfp4.json"
LOCAL_DATASET_ROOTS = (
    Path.home() / "datasets/publish/ptq-calibration-corpus",
    Path.home() / "RESMP-DEV/hf-dataset-card-staging/ptq-calibration-corpus",
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, check=True, **kwargs)


def _ssh(host: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            host,
            "bash",
            "-lc",
            shlex.quote(command),
        ],
        capture_output=capture,
    )


def _api_json(kind: str, repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
    if kind == "dataset":
        url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def verify_hub(campaign: Campaign) -> dict[str, Any]:
    expected_model = campaign.data["model"]["revision"]
    expected_dataset = campaign.data["dataset"]["revision"]
    model = _api_json("model", campaign.data["model"]["id"], expected_model)
    dataset = _api_json("dataset", campaign.data["dataset"]["id"], expected_dataset)
    if model.get("sha") != expected_model:
        raise CampaignError(
            f"model revision moved: pinned {expected_model}, live {model.get('sha')}"
        )
    if dataset.get("sha") != expected_dataset:
        raise CampaignError(
            f"dataset revision moved: pinned {expected_dataset}, live {dataset.get('sha')}"
        )
    result = {
        "model": {
            "id": model.get("id"),
            "sha": model.get("sha"),
            "private": model.get("private"),
            "gated": model.get("gated"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "sha": dataset.get("sha"),
            "private": dataset.get("private"),
            "gated": dataset.get("gated"),
        },
    }
    if hybrid := campaign.data.get("hybrid"):
        source = hybrid["source"]
        live_source = _api_json("model", source["id"], source["revision"])
        if live_source.get("sha") != source["revision"]:
            raise CampaignError(
                "hybrid source revision moved: pinned "
                f"{source['revision']}, live {live_source.get('sha')}"
            )
        result["hybrid_source"] = {
            "id": live_source.get("id"),
            "sha": live_source.get("sha"),
            "private": live_source.get("private"),
            "gated": live_source.get("gated"),
        }
    return result


def find_corpus(campaign: Campaign) -> Path:
    candidates: list[Path] = []
    if override := os.environ.get("PTQ_CALIBRATION_CORPUS"):
        candidates.append(Path(override).expanduser())
    filename = campaign.data["dataset"]["filename"]
    candidates.extend(root / filename for root in LOCAL_DATASET_ROOTS)
    expected = campaign.data["dataset"]["sha256"]
    mismatches: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        actual = sha256_file(candidate)
        if actual == expected:
            return candidate.resolve()
        mismatches.append(f"{candidate}={actual}")
    detail = f"; mismatches: {', '.join(mismatches)}" if mismatches else ""
    raise CampaignError(
        f"pinned calibration artifact {filename} is not available locally; "
        "set PTQ_CALIBRATION_CORPUS" + detail
    )


def inspect_local_model(campaign: Campaign, path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise CampaignError(f"incomplete model checkout at {path}: missing config or index")
    config = json.loads(config_path.read_text())
    expected_type = campaign.data["model"].get("model_type")
    if config.get("model_type") != expected_type:
        raise CampaignError(
            f"wrong model at {path}: expected type {expected_type}, got {config.get('model_type')}"
        )
    index = json.loads(index_path.read_text())
    total_size = int(index.get("metadata", {}).get("total_size", -1))
    expected_size = campaign.data["model"]["weight_bytes"]
    if total_size != expected_size:
        raise CampaignError(
            f"wrong model index at {path}: expected {expected_size} tensor bytes, got {total_size}"
        )
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise CampaignError(
            f"incomplete model checkout at {path}: missing {len(missing)} shard(s), "
            f"including {missing[0]}"
        )
    truncated = []
    for name in shards:
        shard = path / name
        expected_file_size = _safetensors_expected_size(shard)
        if shard.stat().st_size != expected_file_size:
            truncated.append(f"{name}={shard.stat().st_size}/{expected_file_size} bytes")
    if truncated:
        raise CampaignError(f"incomplete model checkout at {path}: truncated shard {truncated[0]}")
    return {
        "path": str(path),
        "model_type": config["model_type"],
        "tensor_bytes": total_size,
        "weight_shards": len(shards),
    }


def _safetensors_expected_size(path: Path) -> int:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise CampaignError(f"invalid safetensors shard {path}: missing header length")
        header_size = int.from_bytes(header_size_raw, "little")
        if header_size <= 0 or header_size > 128 * 1024 * 1024:
            raise CampaignError(f"invalid safetensors shard {path}: bad header length")
        header_raw = handle.read(header_size)
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid safetensors shard {path}: malformed header") from exc
    ends = [
        value["data_offsets"][1]
        for key, value in header.items()
        if key != "__metadata__" and "data_offsets" in value
    ]
    if not ends:
        raise CampaignError(f"invalid safetensors shard {path}: no tensors")
    return 8 + header_size + max(ends)


def find_local_model(
    campaign: Campaign, explicit: str | None = None
) -> tuple[Path, dict[str, Any]]:
    candidates = [Path(explicit).expanduser()] if explicit else []
    model = campaign.data["model"]
    repo_slug = model["id"].replace("/", "-")
    cache_slug = model["id"].replace("/", "--")
    for prefix_length in (7, 8, 12, 40):
        candidates.append(
            Path.home() / "models" / f"{repo_slug}-{model['revision'][:prefix_length]}"
        )
    candidates.extend(
        (
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--{cache_slug}"
            / "snapshots"
            / model["revision"],
        )
    )
    errors = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
            return resolved, inspect_local_model(campaign, resolved)
        except CampaignError as exc:
            errors.append(str(exc))
            if explicit:
                raise
    detail = f"; {'; '.join(errors)}" if errors else ""
    raise CampaignError(
        "no complete local checkout of the pinned model was found; pass an explicit model path"
        + detail
    )


def probe_remote(campaign: Campaign, host: str) -> dict[str, Any]:
    remote_root = campaign.remote_root
    script = r"""
set -eu
root="$1"
gpu_rows="$(nvidia-smi \
  --query-gpu=index,name,memory.total,memory.free,driver_version \
  --format=csv,noheader,nounits)"
python3 - "$root" "$gpu_rows" <<'PY'
import json, os, shutil, socket, subprocess, sys
root, rows = sys.argv[1], sys.argv[2]
expanded = os.path.expanduser(root)
probe_path = expanded
while not os.path.exists(probe_path):
    parent = os.path.dirname(probe_path)
    if parent == probe_path:
        break
    probe_path = parent
usage = shutil.disk_usage(probe_path)
gpus = []
for row in rows.splitlines():
    index, name, memory, free, driver = [part.strip() for part in row.split(',', 4)]
    gpus.append({
        "index": int(index),
        "name": name,
        "memory_mib": int(memory),
        "memory_free_mib": int(free),
        "driver": driver,
    })
print(json.dumps({
    "hostname": socket.gethostname(),
    "gpus": gpus,
    "disk_free_bytes": usage.free,
    "uv": shutil.which("uv"),
    "python": sys.version.split()[0],
    "remote_root": expanded,
}))
PY
"""
    command = f"bash -s -- {shlex.quote(remote_root)}"
    result = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            host,
            command,
        ],
        input=script,
        capture_output=True,
    )
    probe = json.loads(result.stdout)
    minimum_vram = campaign.data["compute"].get("minimum_vram_mib", 0)
    eligible = [gpu for gpu in probe["gpus"] if gpu["memory_mib"] >= minimum_vram]
    probe["requirements"] = {
        "minimum_vram_mib": minimum_vram,
        "eligible_gpu_indices": [gpu["index"] for gpu in eligible],
        "satisfied": bool(eligible),
    }
    return probe


def _receipt_path(campaign: Campaign, name: str) -> Path:
    path = ROOT / "runs" / campaign.name / "receipts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_receipt(campaign: Campaign, name: str, payload: dict[str, Any]) -> Path:
    path = _receipt_path(campaign, name)
    body = {"recorded_at": _now(), "campaign": campaign.name, **payload}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def _require_satisfied(probe: dict[str, Any], allow: bool) -> None:
    if probe["requirements"]["satisfied"] or allow:
        return
    required = probe["requirements"]
    raise CampaignError(
        "CUDA host has no eligible NVIDIA GPU with at least "
        f"{required['minimum_vram_mib']} MiB VRAM"
    )


def eligible_gpus(campaign: Campaign, probe: dict[str, Any]) -> list[dict[str, Any]]:
    minimum_vram = campaign.data["compute"].get("minimum_vram_mib", 0)
    return [gpu for gpu in probe["gpus"] if gpu["memory_mib"] >= minimum_vram]


def solver_vram_gb(campaign: Campaign, gpus: list[dict[str, Any]]) -> list[float]:
    if not gpus:
        raise CampaignError("no CUDA GPUs were detected")
    quant = campaign.data["calibration"]
    fraction = quant.get("vram_fraction", 0.9)
    reserve = quant.get("vram_reserve_gib", 3)
    budgets: list[float] = []
    for gpu in gpus:
        total_gib = gpu["memory_mib"] / 1024
        free_gib = gpu.get("memory_free_mib", gpu["memory_mib"]) / 1024
        budgets.append(round(max(0.1, min(total_gib * fraction, free_gib) - reserve), 1))
    return budgets


def stage_a_argv(
    campaign: Campaign,
    gpu_count: int | list[int],
    vram_gb: float | list[float] | None = None,
) -> list[str]:
    root = campaign.remote_root.rstrip("/")
    quant = campaign.data["calibration"]
    indices = range(gpu_count) if isinstance(gpu_count, int) else gpu_count
    devices = ",".join(f"cuda:{index}" for index in indices)
    if isinstance(vram_gb, list):
        vram_arg = ",".join(str(value) for value in vram_gb)
    else:
        vram_arg = str(vram_gb if vram_gb is not None else 4.0)
    calibration_filename = campaign.data["dataset"]["filename"]
    calibration_option = (
        "--calibration-tokens" if calibration_filename.endswith(".npy") else "--dataset"
    )
    remote_calibration = (
        f"{root}/inputs/calibration.npy"
        if calibration_filename.endswith(".npy")
        else f"{root}/inputs/calibration.txt"
    )
    argv = [
        f"{root}/engine/.venv/bin/python",
        "-m",
        "mlx_gptq.calibrate",
        "--model",
        campaign.remote_model_dir,
        "--output",
        f"{root}/artifacts",
        calibration_option,
        remote_calibration,
        "--nsamples",
        str(quant["nsamples"]),
        "--seqlen",
        str(quant["seqlen"]),
        "--batch-size",
        str(quant["batch_size"]),
        "--mode",
        quant["mode"],
        "--mxfp-algorithm",
        quant["mxfp_algorithm"],
        "--bits",
        str(quant["bits"]),
        "--group-size",
        str(quant["group_size"]),
        "--clip",
        quant["clip"],
        "--damp",
        str(quant["damp"]),
        "--dtype",
        quant["dtype"],
        "--devices",
        devices,
        "--vram-gb",
        vram_arg,
        "--layers-attr",
        campaign.data["model"]["layers_attr"],
    ]
    if prefix := campaign.data["model"].get("checkpoint_model_prefix"):
        argv.extend(("--checkpoint-model-prefix", prefix))
    if prefix := campaign.data["model"].get("artifact_layers_prefix"):
        argv.extend(("--artifact-layers-prefix", prefix))
    argv.extend(("--seed", str(quant["seed"])))
    return argv


def remote_hfd_download_script(campaign: Campaign) -> str:
    model = campaign.data["model"]
    model_dir = campaign.remote_model_dir
    return f"""
set -eu
export HF_HUB_DISABLE_XET=1
hfd_bin="$(command -v hfd || true)"
if [ -z "$hfd_bin" ] && [ -x "$HOME/.local/bin/hfd" ]; then
  hfd_bin="$HOME/.local/bin/hfd"
fi
if [ -z "$hfd_bin" ]; then
  echo 'hfd CLI is required on the CUDA host' >&2
  exit 127
fi
"$hfd_bin" download {shlex.quote(model["id"])} \
  --revision {shlex.quote(model["revision"])} \
  --output {shlex.quote(model_dir)} \
  --backend aria2 --verify full
"$hfd_bin" verify {shlex.quote(model["id"])} \
  --revision {shlex.quote(model["revision"])} \
  --output {shlex.quote(model_dir)} \
  --mode full
"""


def cmd_verify(campaign: Campaign, _args: argparse.Namespace) -> int:
    corpus = find_corpus(campaign)
    hub = verify_hub(campaign)
    receipt = _write_receipt(
        campaign,
        "verify",
        {
            "hub": hub,
            "corpus": {
                "path": str(corpus),
                "bytes": corpus.stat().st_size,
                "sha256": sha256_file(corpus),
            },
            "engine": campaign.data["engine"],
        },
    )
    print(receipt)
    return 0


def cmd_probe(campaign: Campaign, args: argparse.Namespace) -> int:
    probe = probe_remote(campaign, args.host)
    receipt = _write_receipt(campaign, f"probe-{args.host}", {"host": args.host, **probe})
    print(json.dumps(probe, indent=2))
    print(f"receipt: {receipt}")
    _require_satisfied(probe, args.allow_underprovisioned)
    return 0


def cmd_prepare(campaign: Campaign, args: argparse.Namespace) -> int:
    corpus = find_corpus(campaign)
    verify_hub(campaign)
    probe = probe_remote(campaign, args.host)
    _require_satisfied(probe, args.allow_underprovisioned)

    root = campaign.remote_root.rstrip("/")
    model_dir = campaign.remote_model_dir
    engine = campaign.data["engine"]
    model = campaign.data["model"]
    setup = f"""
set -eu
mkdir -p {shlex.quote(root)}/inputs {shlex.quote(model_dir)}
if [ ! -d {shlex.quote(root)}/engine/.git ]; then
  git clone {shlex.quote(engine["repository"])} {shlex.quote(root)}/engine
fi
cd {shlex.quote(root)}/engine
test -z "$(git status --porcelain)"
git fetch origin {shlex.quote(engine["commit"])}
git checkout --detach {shlex.quote(engine["commit"])}
uv sync --frozen --python {shlex.quote(engine["python"])}
uv pip install --python .venv/bin/python accelerate=={shlex.quote(engine["accelerate"])}
"""
    _ssh(args.host, setup)
    remote_calibration = (
        f"{root}/inputs/calibration.npy"
        if corpus.suffix == ".npy"
        else f"{root}/inputs/calibration.txt"
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--partial",
            "--inplace",
            str(corpus),
            f"{args.host}:{remote_calibration}",
        ]
    )
    local_model = None
    local_model_info = None
    try:
        local_model, local_model_info = find_local_model(campaign, args.local_model)
    except CampaignError:
        if args.local_model:
            raise

    if local_model is not None:
        assert local_model_info is not None
        _run(
            [
                "rsync",
                "-a",
                "--checksum",
                "--partial",
                "--inplace",
                "--exclude",
                ".cache/",
                f"{local_model}/",
                f"{args.host}:{model_dir}/",
            ]
        )
        model_source = {"kind": "local-rsync", **local_model_info}
    else:
        download = (
            remote_hfd_download_script(campaign)
            + f"""
sha256sum {shlex.quote(remote_calibration)}
test -f {shlex.quote(model_dir)}/model.safetensors.index.json
"""
        )
        _ssh(args.host, download)
        model_source = {
            "kind": "remote-hfd-download",
            "id": model["id"],
            "revision": model["revision"],
            "verification": "full",
        }
    receipt = _write_receipt(
        campaign,
        f"prepare-{args.host}",
        {
            "host": args.host,
            "probe": probe,
            "remote_root": root,
            "remote_model_dir": model_dir,
            "model_source": model_source,
        },
    )
    print(receipt)
    return 0


def cmd_start(campaign: Campaign, args: argparse.Namespace) -> int:
    probe = probe_remote(campaign, args.host)
    _require_satisfied(probe, args.allow_underprovisioned)
    selected_gpus = eligible_gpus(campaign, probe)
    if not selected_gpus and args.allow_underprovisioned:
        selected_gpus = probe["gpus"]
    argv = stage_a_argv(
        campaign,
        [gpu["index"] for gpu in selected_gpus],
        vram_gb=solver_vram_gb(campaign, selected_gpus),
    )
    root = campaign.remote_root.rstrip("/")
    model_dir = campaign.remote_model_dir
    calibration_filename = campaign.data["dataset"]["filename"]
    remote_calibration = (
        f"{root}/inputs/calibration.npy"
        if calibration_filename.endswith(".npy")
        else f"{root}/inputs/calibration.txt"
    )
    command = shlex.join(argv)
    start = f"""
set -eu
test -f {shlex.quote(model_dir)}/model.safetensors.index.json
test -f {shlex.quote(remote_calibration)}
if [ -f {shlex.quote(root)}/run.pid ] && \
  kill -0 "$(cat {shlex.quote(root)}/run.pid)" 2>/dev/null; then
  echo 'calibration is already running' >&2
  exit 3
fi
nohup {command} > {shlex.quote(root)}/stage-a.log 2>&1 < /dev/null &
echo $! > {shlex.quote(root)}/run.pid
cat {shlex.quote(root)}/run.pid
"""
    result = _ssh(args.host, start, capture=True)
    receipt = _write_receipt(
        campaign,
        f"start-{args.host}",
        {"host": args.host, "pid": int(result.stdout.strip()), "argv": argv, "probe": probe},
    )
    print(receipt)
    return 0


def cmd_status(campaign: Campaign, args: argparse.Namespace) -> int:
    root = campaign.remote_root.rstrip("/")
    status = f"""
set -u
pid=''
state='not-started'
if [ -f {shlex.quote(root)}/run.pid ]; then
  pid="$(cat {shlex.quote(root)}/run.pid)"
  if kill -0 "$pid" 2>/dev/null; then state='running'; else state='stopped'; fi
fi
printf 'state=%s pid=%s\n' "$state" "$pid"
if [ -f {shlex.quote(root)}/artifacts/manifest.json ]; then
  python3 - {shlex.quote(root)}/artifacts/manifest.json <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print('layers_done=' + str(len(d.get('layers_done', []))))
print('tensor_count=' + str(len(d.get('tensors', {{}}))))
PY
fi
if [ -f {shlex.quote(root)}/stage-a.log ]; then
  tail -n {int(args.lines)} {shlex.quote(root)}/stage-a.log
fi
"""
    result = _ssh(args.host, status, capture=True)
    print(result.stdout, end="")
    return 0


def cmd_fetch(campaign: Campaign, args: argparse.Namespace) -> int:
    root = campaign.remote_root.rstrip("/")
    destination = ROOT / "runs" / campaign.name / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-a",
            "--partial",
            "--checksum",
            f"{args.host}:{root}/artifacts/",
            f"{destination}/",
        ]
    )
    manifest = destination / "manifest.json"
    if not manifest.is_file():
        raise CampaignError(f"fetched artifacts have no manifest: {manifest}")
    payload = json.loads(manifest.read_text())
    receipt = _write_receipt(
        campaign,
        f"fetch-{args.host}",
        {
            "host": args.host,
            "destination": str(destination),
            "manifest_sha256": sha256_file(manifest),
            "layers_done": len(payload.get("layers_done", [])),
            "tensor_count": len(payload.get("tensors", {})),
        },
    )
    print(receipt)
    return 0


def _default_overlay_path(campaign: Campaign) -> Path:
    hybrid = campaign.data.get("hybrid")
    if hybrid is None:
        raise CampaignError("campaign has no hybrid tensor-overlay configuration")
    filename = hybrid.get("overlay_name", "bf16-overlay.safetensors")
    return ROOT / "runs" / campaign.name / "overlays" / filename


def cmd_fetch_overlay(campaign: Campaign, args: argparse.Namespace) -> int:
    hybrid = campaign.data.get("hybrid")
    if hybrid is None:
        raise CampaignError("campaign has no hybrid tensor-overlay configuration")
    output = Path(args.output).expanduser() if args.output else _default_overlay_path(campaign)
    source = hybrid["source"]
    result = fetch_overlay(source["id"], source["revision"], hybrid["match"], output)
    receipt = _write_receipt(campaign, "fetch-overlay", result)
    print(receipt)
    return 0


def cmd_apply_overlay(campaign: Campaign, args: argparse.Namespace) -> int:
    hybrid = campaign.data.get("hybrid")
    if hybrid is None:
        raise CampaignError("campaign has no hybrid tensor-overlay configuration")
    overlay = Path(args.overlay).expanduser() if args.overlay else _default_overlay_path(campaign)
    result = apply_overlay(
        Path(args.model),
        overlay,
        hybrid["source_prefix"],
        hybrid["target_prefix"],
    )
    receipt = _write_receipt(campaign, "apply-overlay", result)
    print(receipt)
    return 0


def cmd_pack(campaign: Campaign, args: argparse.Namespace) -> int:
    engine = Path(args.engine).expanduser().resolve()
    expected_commit = campaign.data["engine"]["commit"]
    actual_commit = _run(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], capture_output=True
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise CampaignError(f"engine is {actual_commit}; campaign pins {expected_commit}")
    python = engine / ".venv" / "bin" / "python"
    if not python.is_file():
        raise CampaignError(
            f"missing {python}; run `uv sync --directory {engine} --frozen --extra mlx` first"
        )
    source_model, model_info = find_local_model(campaign, args.model)
    calib = ROOT / "runs" / campaign.name / "calibration"
    output_value = args.output or f"~/models/{campaign.data['output']['mlx_model_name']}"
    output = Path(output_value).expanduser().resolve()
    argv = [
        str(python),
        "-m",
        "mlx_gptq.pack",
        "--hf-path",
        str(source_model),
        "--calib",
        str(calib),
        "--mlx-path",
        str(output),
    ]
    if keep_bf16 := campaign.data.get("packing", {}).get("keep_bf16_regex"):
        argv.extend(("--router-skip", keep_bf16))
    if not args.overlay:
        argv.append("--verify")
    _run(argv, cwd=engine)
    overlay_result = None
    if args.overlay:
        hybrid = campaign.data.get("hybrid")
        if hybrid is None:
            raise CampaignError("--overlay requires campaign hybrid configuration")
        overlay_result = apply_overlay(
            output,
            Path(args.overlay),
            hybrid["source_prefix"],
            hybrid["target_prefix"],
        )
        _run(
            [
                str(python),
                "-c",
                "import sys; from mlx_gptq.pack import verify; verify(sys.argv[1])",
                str(output),
            ],
            cwd=engine,
        )
    receipt = _write_receipt(
        campaign,
        "pack",
        {
            "source_model": model_info,
            "output": str(output),
            "engine_commit": actual_commit,
            "keep_bf16_regex": campaign.data.get("packing", {}).get("keep_bf16_regex"),
            "overlay": overlay_result,
        },
    )
    print(receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="verify pinned Hub revisions and the local corpus")

    for name in ("probe", "prepare", "start"):
        command = sub.add_parser(name)
        command.add_argument("--host", required=True)
        command.add_argument("--allow-underprovisioned", action="store_true")
        if name == "prepare":
            command.add_argument("--local-model")

    status = sub.add_parser("status")
    status.add_argument("--host", required=True)
    status.add_argument("--lines", type=int, default=40)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--host", required=True)

    fetch_overlay_parser = sub.add_parser(
        "fetch-overlay", help="fetch pinned BF16 tensors selected by the campaign"
    )
    fetch_overlay_parser.add_argument("--output")

    apply_overlay_parser = sub.add_parser(
        "apply-overlay", help="apply a fetched BF16 tensor overlay to a packed MLX model"
    )
    apply_overlay_parser.add_argument("--model", required=True)
    apply_overlay_parser.add_argument("--overlay")

    pack = sub.add_parser("pack")
    pack.add_argument("--engine", default="~/mlx-gptq")
    pack.add_argument("--model")
    pack.add_argument("--output")
    pack.add_argument("--overlay")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        handlers = {
            "verify": cmd_verify,
            "probe": cmd_probe,
            "prepare": cmd_prepare,
            "start": cmd_start,
            "status": cmd_status,
            "fetch": cmd_fetch,
            "fetch-overlay": cmd_fetch_overlay,
            "apply-overlay": cmd_apply_overlay,
            "pack": cmd_pack,
        }
        return handlers[args.command](campaign, args)
    except (CampaignError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
