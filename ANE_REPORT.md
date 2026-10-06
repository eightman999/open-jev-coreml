# ANE report — JevLite on Core ML / Apple Neural Engine

What ran on the Neural Engine path, what fell back, what it costs in
accuracy, and what to try next. All numbers are measured on the machine
below; nothing is extrapolated.

## Machine / environment

| | |
|---|---|
| machine | Mac14,6 — Apple M2 Max, 32 GB |
| macOS | 26.6.2 (25G83), arm64 |
| python | 3.12.11 (`.venv`, via uv — system Python 3.14 was not trusted for coremltools) |
| torch | 2.14.0 (coremltools 9 warns it is untested > 2.7; export works) |
| transformers | 5.17.0 |
| coremltools | 9.0 |
| Xcode | 26.6 (17F113) |
| recorded | `artifacts/environment.json` |

## Base commit / model

- fork: `eightman999/open-jev-typed-decision-engine`, branch `feat/coreml-ane`
- upstream base: `78d3b3a171f24d8d9a8dea18e027f9d3373fda45`
- checkpoint: `jevlite.pt` — ModernBERT-base + `Linear(d,1)` token head,
  trained 20 epochs on MPS (`02_train.py`), **best val accuracy 0.6859**,
  **test accuracy 0.6430, ECE 0.0536** on the official 400-case split
  (`04_eval.py`). Temperatures stayed at 1.0 — `03_calibrate.py` found the
  soft-target loss already calibrated (same outcome upstream reported).

## What was exported

`export_coreml.py` exports `token_logits [1, S]` — encoder + score head at
every position. The dynamic part (label-position gather, per-question
softmax, temperature, typed output) stays in `coreml_backend.py` on CPU, so
the schema remains data. Nothing about questions or labels is in the graph.

| setting | value |
|---|---|
| format | ML Program (`.mlpackage`) |
| precision | fp16 — **requires `--min-macos 26`**, see failed experiments |
| inputs | `input_ids`, `attention_mask`, int32 `[1, S]` |
| sequence buckets | 256 / 512 / 1024 (smallest covering bucket wins; >1024 raises — the state-truncation policy lives in `td_data.encode`, unchanged) |
| package size | ~300 MB each |
| conversion path | `torch.export` → `run_decompositions` → `ct.convert` |

Two op gaps in coremltools 9 had to be patched in `export_coreml.py`
(transformers 5.x masking emits them): `new_ones` (implemented via
`mb.fill` + dtype cast) and the EXIR `__and__`/`__or__` dunder ops
(registered directly — `register_torch_op` misrejects dunder names as
inplace).

## Compute units

`cpu_only`, `cpu_and_ne`, `cpu_and_gpu`, `all` are all selectable
(`CoreMLEngine(compute_units=...)`, `--compute-units` on the CLIs).
`cpu_and_ne` excludes the GPU — that is the ANE-candidate path, and it is
what this report calls "the ANE path". Per-op placement was **not**
verified with an Xcode performance report; the claim "ANE participates"
rests on `cpu_and_ne` being consistently faster than `cpu_only` with
identical math.

## Parity — PyTorch reference vs Core ML

Full test split, 400 cases / 2000 decisions, `cpu_and_ne`
(`artifacts/parity_full.json`):

| metric | value | target |
|---|---|---|
| decision agreement | **0.9965** | ≥ 0.99 ✓ |
| coreml accuracy | **0.6445** | — |
| torch accuracy | 0.6430 | — |
| accuracy regression | **−0.15 pp** (none) | ≤ 0.5 pp ✓ |
| confidence MAE | **0.0024** | ≤ 0.02 ✓ |
| max \|Δlogit\| | 0.1504 | — |
| mean \|Δlogit\| | 0.0109 | — |

50-row breakdown by compute unit (`artifacts/parity_report.json`):

| compute unit | max Δlogit | agreement | conf MAE |
|---|---|---|---|
| cpu_and_ne | 0.0396 | 0.996 | 0.0018 |
| all | 0.0526 | 1.000 | 0.0021 |
| cpu_only | 0.1773 | 0.980 | 0.0056 |

Note the ordering: `cpu_only` fp16 is the *worst* unit — the ANE fp16
kernels land closer to fp32 than Core ML's own fp16 CPU kernels do.

fp32 export (`--precision fp32`, `artifacts/coreml-fp32/`) is **exact**:
max Δlogit 0.0000, agreement 1.0000 on 25 rows. It is the correctness
fallback — it cannot use the ANE (fp16 only), but it proves the graph.

## Latency — `artifacts/benchmark.json`

batch 1, 20 warmup + 100 measured iterations, real encoded inputs, machine
idle. Full table in `BENCHMARK.md`.

| seq | torch cpu | torch mps | cml cpu_only | cml cpu_and_ne | cml all |
|---|---|---|---|---|---|
| 256 | 71.5 | 13.4 | 32.8 | **13.2** | 14.1 |
| 512 | 123.8 | 22.9 | 64.2 | **37.1** | 37.3 |
| 1024 | 253.7 | 47.4 | 169.0 | **111.1** | 110.4 |
| | | | | (median ms) | |

- `cpu_and_ne` beats `cpu_only` at every length: **2.5× / 1.7× / 1.5×**.
- `cpu_and_ne` beats PyTorch-CPU by **5.4× / 3.3× / 2.3×**.
- `all` ≈ `cpu_and_ne` — the GPU adds nothing here; do not read `all` as
  "ANE speed".
- PyTorch-MPS still wins at 512/1024. The ANE path is the low-power and
  CPU-freeing option, not the raw-latency winner on this model at long
  sequences.

## Known CPU fallbacks

- **fp32 packages run entirely on CPU/GPU** — the ANE is fp16-only.
- Inside `cpu_and_ne` fp16, Core ML decides placement per op; everything
  that does not fit an ANE kernel (e.g. the int32 embedding gather, any
  op without an ANE path) runs on CPU. A per-op placement trace was not
  produced — that is the remaining measurement gap.
- The decision head tail (gather, grouped softmax, temperature, output
  shaping) is CPU by design — a few µs of numpy per call.

## Failed experiments / traps hit

1. **fp16 at `minimum_deployment_target=macOS15` destroys the model.**
   First export targeted macOS15: max Δlogit **5.54**, mean **2.34**,
   agreement **0.34**, accuracy 0.36 vs 0.72. torch fp16 on CPU is fine
   (max err 0.019) and no intermediate exceeds 612 in magnitude — the
   macOS15-target kernels accumulate differently, not just round weights.
   **`--min-macos 26` fixed it completely** (max err 0.04). Suspect an
   older-target fp16 matmul/attention path with fp16 accumulation.
   Do not ship macOS15-targeted fp16 packages.
2. **`eager` attention does not convert.** Forcing
   `config._attn_implementation="eager"` makes transformers emit a cast
   pattern coremltools 9 rejects (`TypeError: only 0-dimensional arrays
   can be converted to Python scalars` in `_int`). The sdpa-derived graph
   is the only working path on this stack.
3. **Weight palettization is not worth it here.** 4-bit kmeans: 299→75 MB
   but max Δlogit 6.30 — destroys the head. 8-bit: ~150 MB, max Δlogit
   0.75 (20× worse than fp16) for a 1 ms latency gain. Bandwidth is not
   the bottleneck; both rejected, fp16 packages ship unchanged.
4. **In-process ANE predict corrupts the host heap (probabilistic).**
   macOS crash reports show an ObjC object graph containing a `PyObject*`
   destroyed on a background dispatch thread
   (`objc_destructInstance` → `_PyObject_Free`, `EXC_BAD_ACCESS` at 0x10),
   after which the next unrelated native call faults (ssl `poll`,
   `pathlib.rglob`). ~1 in 15 predicts under memory pressure; observed
   during pytest runs concurrent with MPS training, and it also killed a
   quiet-machine test run once. Since `feat/coreml-ane-worker-bench`,
   **`coreml_worker.py` keeps the engine in a child process in production
   too** (`DecisionEngine`/`05_serve.py` default `coreml_mode="worker"`,
   respawn-once + periodic recycle; `inprocess` is diagnostic-only).
   `CoreMLEngine._keepalive` still retains predict outputs inside the
   worker — deallocating them is what trips the bug — bounded by worker
   lifetime and `recycle_every` (10k decisions ≈ tens of MB).
5. transformers 5.17 + coremltools 9 needed three missing-op registrations
   (above). torch 2.14 is untested by coremltools 9 but converted cleanly.
6. Copying predict outputs off the ObjC-bridged buffer and retaining the
   output dict were both tried for the heap issue — neither alone was
   sufficient; isolation is what works.

## Remaining bottlenecks

- **512/1024 latency vs MPS**: ANE path loses to torch-MPS at long
  sequences (37 vs 23 ms @512, 111 vs 47 ms @1024). If the win exists it
  is in attention implementation/layout — needs an operator-placement
  trace first.
- **cpu_only fp16 parity** (0.98 agreement) sits just under the 99% bar;
  users needing exact CPU parity should use fp32 packages.
- **Package size** (~300 MB each; ~900 MB for the bucket set). Weight
  palettization would shrink it — deliberately not done yet because
  accuracy impact must be measured first.
- No batching support: all numbers are batch=1.

## Jev competition benchmark

The worker-isolated backend was benchmarked against the historical Jev
judging workload used to build the novllm dataset (11,842 paired verdicts
from the longspan-resplit judge queues). Results - paired agreement,
confidence routing, latency/throughput per backend, and the honest
limitations - are in **[JEV_COMPETITION.md](JEV_COMPETITION.md)**.

## Best next experiment

1. Operator-placement trace (Xcode Core ML performance report or
   `MLModelConfiguration` profiling) on `cpu_and_ne` @1024 — find which
   ops fall back to CPU and whether attention is the fp16/ANE bottleneck.
   Until that exists, any layout optimization is guessing.
2. If a newer coremltools adds per-op fp32 overrides, keep just
   LayerNorm/softmax fp32 inside a macOS15-targeted fp16 build — would
   tell us whether the old-target failure is accumulation in norms.
3. Batch>1 packages if throughput ever matters; all current numbers are
   batch=1.
