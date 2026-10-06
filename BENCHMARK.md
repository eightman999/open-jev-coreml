# Benchmark — JevLite: PyTorch vs Core ML on Apple Silicon

Reproduce:

```bash
python benchmark_coreml.py --ckpt jevlite.pt --iters 100 --out artifacts/benchmark.json
```

Method: one `token_logits` forward pass at a fixed `[1, S]` input, batch 1.
Input is a real encoded test row padded to the bucket, identical bytes for
every backend. **20 warmup iterations, then 100 measured**, wall-clock per
call. Machine otherwise idle. Full data: `artifacts/benchmark.json`.

Environment: Mac14,6 Apple M2 Max 32 GB, macOS 26.6.2, torch 2.14.0,
coremltools 9.0, packages fp16 ML Program @ min-macOS 26.

## Median latency (ms)

| backend | seq 256 | seq 512 | seq 1024 |
|---|---|---|---|
| torch cpu | 71.5 | 123.8 | 253.7 |
| torch mps | 13.4 | 22.9 | 47.4 |
| coreml cpu_only | 32.8 | 64.2 | 169.0 |
| **coreml cpu_and_ne** | **13.2** | **37.1** | **111.1** |
| coreml all | 14.1 | 37.3 | 110.4 |

## Detail (median / p90 / p95 / mean ms, req/s, tok/s)

### seq 256

| backend | median | p90 | p95 | mean | req/s | tok/s |
|---|---|---|---|---|---|---|
| torch_cpu | 71.5 | 73.8 | 75.8 | 72.8 | 14.0 | 3,579 |
| torch_mps | 13.4 | 13.7 | 13.9 | 13.5 | 74.6 | 19,101 |
| coreml_cpu_only | 32.8 | 33.5 | 33.8 | 33.0 | 30.5 | 7,811 |
| coreml_cpu_and_ne | 13.2 | 13.2 | 13.2 | 13.2 | 76.0 | 19,454 |
| coreml_all | 14.1 | 14.3 | 14.4 | 14.2 | 71.1 | 18,198 |

### seq 512

| backend | median | p90 | p95 | mean | req/s | tok/s |
|---|---|---|---|---|---|---|
| torch_cpu | 123.8 | 129.6 | 136.8 | 125.8 | 8.1 | 4,136 |
| torch_mps | 22.9 | 23.2 | 23.3 | 22.9 | 43.7 | 22,355 |
| coreml_cpu_only | 64.2 | 65.4 | 65.7 | 64.4 | 15.6 | 7,978 |
| coreml_cpu_and_ne | 37.1 | 37.4 | 38.2 | 37.2 | 27.0 | 13,809 |
| coreml_all | 37.3 | 38.6 | 38.7 | 37.6 | 26.8 | 13,719 |

### seq 1024

| backend | median | p90 | p95 | mean | req/s | tok/s |
|---|---|---|---|---|---|---|
| torch_cpu | 253.7 | 274.5 | 280.0 | 255.9 | 3.9 | 4,037 |
| torch_mps | 47.4 | 48.2 | 48.6 | 47.6 | 21.1 | 21,585 |
| coreml_cpu_only | 169.0 | 179.7 | 185.2 | 170.9 | 5.9 | 6,059 |
| coreml_cpu_and_ne | 111.1 | 112.1 | 113.4 | 111.3 | 9.0 | 9,215 |
| coreml_all | 110.4 | 110.8 | 110.9 | 110.3 | 9.1 | 9,275 |

## Load time / package size

| | seconds |
|---|---|
| torch_cpu | 2.8 |
| torch_mps | 2.0 |
| coreml_cpu_only (3 pkgs) | 3.6 |
| coreml_all (3 pkgs) | 24.5 |
| coreml_cpu_and_ne (3 pkgs) | **29.4** |

Package size: ~299 MB per bucket (~899 MB for the set), fp16 weights.

`cpu_and_ne` load is the expensive one — first-load ANE compilation. That
cost is one-time per process; it is why `CoreMLEngine` loads eagerly in
`__init__` so `load_time_s` is honest.

## Reading the numbers

- `cpu_and_ne` < `cpu_only` at every sequence length → the GPU-disabled
  path is genuinely faster than pure CPU: the Neural Engine is doing work.
- `all` ≈ `cpu_and_ne` — the GPU does not help this model here. `all` is
  reported for completeness, **not** as ANE performance.
- PyTorch-MPS beats the ANE path at 512/1024. The honest summary: ANE is
  the fastest non-GPU path (2–5× over CPU) and ties MPS at seq 256, but
  for raw latency at long sequences this machine's GPU wins.
- Core ML `cpu_only` is ~2× faster than torch CPU at every length — Core
  ML's CPU kernels are worth having even without ANE.
- tok/s uses the bucket length (padded), matching how the model is billed
  internally.
