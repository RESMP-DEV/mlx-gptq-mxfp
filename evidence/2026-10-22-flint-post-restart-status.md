# Flint post-restart status receipt

Date: 2026-10-22
Agent: Codex

## Verified state

- `RESMP-DEV/mlx-lm` main is `e01954dd3999efd0850e73ebf765287f5e7311dc`
  and matches `origin/main`. It contains Flint support at `428b496` and the
  fused expert-gate/up-gather optimization at `e01954d`.
- `RESMP-DEV/mlx-stack` main is `852d20b3b7baeed11baef321269959073d5e0307`
  and matches `origin/main`.
- Post-restart unit checks:
  - `PYTHONPATH=. uv run --no-project --with 'mlx==0.32.4' --with pytest
    --with numpy python -m pytest -q tests/test_flint.py` passed 3 tests.
  - `.venv/bin/python -m pytest -q tests/test_flint_torch.py` passed 4 tests.
- B550 completed all 49 Flint calibration layers in 44.9 minutes. Its
  validation reports `status: verified`, 36,439 tensors, and no failures.
- The Mac and B550 calibration manifests both have SHA-256
  `581727ff80fd18b789b48a2cd19e74d78a646a33403b95f0393a640121324ad7`;
  validation receipts both have SHA-256
  `5cdf3127ce3a5158fca7b72e578e7b0b80044da469d1ea751a6fb71b28c73473`.
  The Mac has all 49 layer artifacts.
- The full packed model is
  `/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4`.
  It has five safetensors shards totaling 22,075,860,615 bytes and a
  1,263-entry weight map. Packing completed through `--verify`; that code
  path loads the full model through MLX-LM and generates 40 tokens.
- No Flint process remains on B550. Its restore receipt reports
  `{"status":"loaded"}`. The GPU process now visible on B550 is the unrelated
  own-Coder `llama-imatrix` campaign and must not be attributed to Flint.

## Restart diagnosis

- macOS booted at 10:29:45 PDT on 2026-10-22.
- The boot diagnostic reports `Boot failure count: 1` and boot faults
  `rst btn_rst,btn_seq_reset target_off_restart`.
- No matching kernel-panic report was found. This proves a button/sequence
  reset path, but it does not identify the original user-interface stall.
- Ten minutes after reboot, load had fallen from the initial post-reboot spike
  to about 7, swap usage was zero, and memory pressure remained healthy.

## Non-claims

- The 40-token pack verification is format and load evidence, not quality
  evidence.
- Holdout perplexity, output parity, serving performance, and Hugging Face
  publication are not complete.
- This receipt does not prove what caused the pre-reset interface stall.
