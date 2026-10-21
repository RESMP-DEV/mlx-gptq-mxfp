# HTTP server backlog hardening

## Claim

The MLX-LM HTTP server was dropping connections under a burst of concurrent
requests because it listened with the stdlib default backlog. The DFlash
runtime now serves with a burst-sized backlog, and the failure is gone:
32-way, 64-way, and 128-way bursts complete with zero errors and zero
retries.

## Root cause

The stdlib server class listens with a small pending-connection queue. On the
Python 3.13 interpreter the services use, that default is 5:

```text
socketserver.TCPServer.request_queue_size = 5
http.server.ThreadingHTTPServer.request_queue_size = 5
```

A burst of concurrent clients that arrives faster than the accept loop
drains the queue gets its excess connections reset by the kernel. The client
sees a broken pipe, an invalid argument, or a connection reset, while the
server records the request as served with status 200. The kernel limit on
this machine is far higher, so this is not a machine limit:

```text
kern.ipc.somaxconn: 2048
ulimit -n: unlimited
```

A controlled experiment on the exact interpreter the services use, with the
server response shape held constant, isolated the variable:

```text
python 3.13.9
stdlib default (backlog=5)         failures= 67/96
hardened (backlog=1024)             failures=  0/96
```

Switching only the backlog, with the protocol version and the response
shape unchanged, removed every failure. A second experiment confirmed the
protocol version is not the cause: HTTP/1.0 and HTTP/1.1 both failed with
backlog 5 and both passed with backlog 128 or 1024.

## Change

`runtime/dflash-mlx/dflash_mlx/server/http_server.py` adds a server class
with `request_queue_size` of 1024, `allow_reuse_address` enabled, and
daemon threads. The class is produced by a factory rather than a module
constant because the MLX-LM server assigns `address_family` on the class
object it is given, and that value must not leak between servers that bind
different address families.

`runtime/dflash-mlx/dflash_mlx/server/runtime.py` passes that class to
`mlx_lm.server._run_http_server`, which already accepts a `server_class`.
The external MLX-LM contract is unchanged.

The DFlash prefix-cache and admission behavior is unchanged. The admission
proxy stays as the second line of defense for the serial lane.

## Live result on the target batch lane

Three consecutive 32-way runs of 128 tokens, with streaming health and
token-ID capture enabled:

| Run | Requests ok | Errors | Retries | Aggregate tok/s | Wall s |
|---|---:|---:|---:|---:|---:|
| 1 | 32 | 0 | 0 | 128.17 | 31.96 |
| 2 | 32 | 0 | 0 | 139.97 | 29.26 |
| 3 | 32 | 0 | 0 | 133.15 | 30.76 |

Before the change, the same workload needed 13 and 12 retries in two of
three runs, and failed 9 and 13 requests outright in the runs before that.

Larger bursts, with the same service:

| Concurrency | Requests ok | Errors | Retries | Aggregate tok/s | Wall s |
|---|---:|---:|---:|---:|---:|
| 64 | 64 | 0 | 0 | 129.35 | 31.66 |
| 128 | 128 | 0 | 0 | 119.88 | 68.34 |

The token-ID digest stayed `a41e620f00de4df4d753a5a24f48b3218ea34c92755abba03c5a6c40299b7491`
in every run, the same value the serial lane produced, so the hardening
changed connection behavior and not output.

## Live result on the resident serial lane

The serial lane runs the same runtime and received the same fix. A 32-way
run of 64 tokens against the admission proxy completed 32 of 32 with zero
errors and zero retries, at 23.39 aggregate tok/s. The same workload
measured 17.16 aggregate tok/s before the change.

## Verification

```text
runtime/dflash-mlx/tests/test_http_server.py: 6 passed
```

The test file covers the server class attributes, the fresh-class-per-call
behavior, the wiring into the serve path, a burst that the hardened class
serves without errors, and a control case that the stdlib backlog fails the
same burst. The control case fails the suite if the stdlib default ever
changes, so the hardening keeps a real reason.

```text
just quality
passed=true
```

## Non-claims

- The backlog value is a burst-size choice, not a measured optimum. The
  kernel limit on this machine is 2048.
- The 128-way result is a single run and the aggregate rate falls with
  concurrency, so it is a robustness data point, not a capacity figure.
- The client-side retry in the acceptance probe is unchanged and still
  present. It is now a safety net rather than the mechanism that makes
  bursts work.
- No upstream MLX-LM source was modified.
