import argparse
import json
from pathlib import Path

from tools.export_dflash2_capture_split import (
    export_split,
    prefix_cycles,
    split_prompts,
)


def test_split_prompts_is_disjoint_and_deterministic():
    first = split_prompts([3, 1, 3, 2, 1], fraction=0.34, seed=17)
    second = split_prompts([3, 1, 3, 2, 1], fraction=0.34, seed=17)

    assert first == second
    train, validation = first
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == [1, 2, 3]


def test_prefix_cycles_keeps_each_prompt_contiguous():
    records = [{"prompt_index": prompt} for prompt in [0] * 8 + [1] * 2]

    selected = prefix_cycles(records, limit=5)
    prompts = [records[index]["prompt_index"] for index in selected]
    prompt_zero = [index for index in selected if index < 8]

    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert prompts.count(0) == 4
    assert prompts.count(1) == 1
    assert prompt_zero == list(range(4))


def test_export_split_copies_only_selected_cycle_files(tmp_path: Path):
    source = tmp_path / "source"
    source_prompts = source / "prompts"
    source_cycles = source / "cycles"
    source_prompts.mkdir(parents=True)
    source_cycles.mkdir()
    prompt_records = [
        {"prompt_index": prompt, "cycles": 2} for prompt in range(4)
    ]
    cycle_records = []
    cycle = 0
    for prompt in range(4):
        (source_prompts / f"{prompt:06d}.safetensors").write_bytes(b"prompt")
        for _ in range(2):
            cycle_dir = source_cycles / f"{cycle:06d}"
            cycle_dir.mkdir()
            (cycle_dir / "tensors.safetensors").write_bytes(b"cycle")
            cycle_records.append({"cycle": cycle, "prompt_index": prompt})
            cycle += 1
    (source / "index.json").write_text(
        json.dumps({"prompts": prompt_records, "cycles": cycle_records})
    )

    args = argparse.Namespace(
        capture_dir=source,
        output_dir=tmp_path / "split",
        validation_fraction=0.25,
        train_cycle_limit=4,
        validation_cycle_limit=2,
        train_prompt_limit=2,
        validation_prompt_limit=1,
        seed=17,
    )
    receipt = export_split(args)

    train_index = json.loads((tmp_path / "split/train/index.json").read_text())
    validation_index = json.loads(
        (tmp_path / "split/validation/index.json").read_text()
    )
    train_prompt_ids = {
        record["prompt_index"] for record in train_index["cycles"]
    }
    validation_prompt_ids = {
        record["prompt_index"] for record in validation_index["cycles"]
    }
    assert receipt["train_cycles"] == 4
    assert receipt["validation_cycles"] == 2
    assert receipt["selected_train_prompts"] == 2
    assert receipt["selected_validation_prompts"] == 1
    assert train_prompt_ids.isdisjoint(validation_prompt_ids)
    assert len(list((tmp_path / "split/train/cycles").iterdir())) == 4
    assert len(list((tmp_path / "split/validation/cycles").iterdir())) == 2
