# Local27 prefill L2 survival receipt

Date: 2026-10-21
Agent: Codex

## Change

The resident DFlash server discarded all prefix-cache work for generation
requests until end-of-generation publication. A client disconnect after a
202,300-token prefill therefore forced the retry to prefill from token zero.
Generation requests now publish their stable prefill frontier to L2 only. L1
continues to reserve itself for the end-of-generation snapshot.

The resident lane also has a KeepAlive metrics watcher. It records request
state, prefill progress, cache outcomes, and memory to
`~/.local/state/local27-mxfp4/watch/metrics-watch.jsonl`.

## Live diagnosis

- Request 8: 202,300 physical prefill tokens, 2,587.14 s, TTFT 2,587.14 s,
  one generated token, `finish_reason: null`.
- Request 12: identical prompt length, cache lookup miss with
  `longest_token_match_len: 3`, then a second cold prefill.
- Cache event source:
  `~/.local/state/local27-mxfp4/diagnostics/cache_events.jsonl`.
- Watcher source:
  `~/.local/state/local27-mxfp4/watch/metrics-watch.jsonl`.

## Validation

- `PYTHONPATH=... pytest -q tests/test_prefill_step_size_runtime.py -k
  'frontier or generation_snapshot_published or generation_request_publishes'`
  passed 6 tests.
- `shellcheck scripts/local27_mxfp4_service.sh` passed.
- `python3 -m py_compile scripts/local27_metrics_watcher.py` passed.
- `plutil -lint scripts/dev.resmp.local27-metrics-watch.plist` passed.
- The complete `test_prefill_step_size_runtime.py` run had 56 passes and one
  unrelated adaptive-block assertion failure
  (`test_adaptive_verify_mode_drops_and_recovers_block_len`; expected one
  reduction, observed two). The focused snapshot tests all passed.
- Live restart validation used an exact tokenizer-derived 9,026-token prompt.
  The first request was cold: 9,026 physical prefill tokens, 67.9417s prefill,
  TTFT 67.9425s. The identical retry was warm: 9,019 restored tokens, seven
  physical tokens, 0.2419s prefill, TTFT 0.2426s, and 0.324s client wall time.
- Before restart, `lsof` showed no established client on ports 8080 or 8081;
   the in-process 202,300-token request was an orphaned prefill.

## Non-claims

- This validates prefix restoration and latency only. It does not prove
  task-output parity.
