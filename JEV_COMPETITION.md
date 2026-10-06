# Jev vs JevLite — the real novllm judging workload

Can the local Core ML / ANE JevLite reproduce (or complement) the hosted
Jev classifier that filtered the novllm dataset? Benchmarked on the
**actual historical judge queues and cached verdicts** — not a synthetic
proxy. Zero API calls; every Jev number comes from the verdict files
recorded during dataset construction.

**TL;DR — not with this checkpoint.** JevLite answers `uncertain` on
~99.8% of the workload while historical Jev split it ~99.8/0.2
noise/content (the resplit set is a filtered second pass — `uncertain`
never survives into these verdicts). Paired agreement is **0.15–0.24%**
across all four local backends. The machinery is proven end-to-end —
47,368 real inferences, zero failures, worker isolation holding, ANE
faster than CPU — but this checkpoint never learned corpus cleaning. The
harness is ready for a checkpoint that was.

## What was compared

### Workload (input/output sha256 recorded in each report's `meta`)

| set | items | paired verdicts | source |
|---|---|---|---|
| longspan resplit judge queues | 11,842 | 11,842 (100%) | `j72-350m-mac/judge/longspan/queue_{350m,600m,1b,1500m}.resplit.jsonl` + `verdicts_*.resplit.jev.jsonl` |
| 1500m ext judge queue (secondary) | 14,046 | 14,046 (100%) | `dataset_1500m/ext/judge_queue.jsonl` + `verdicts_1500m.jev.jsonl` |

The resplit set is the secondary judge pass: long-span candidates the
first pass flagged were re-split and re-judged. Every item has a paired
historical Jev verdict (`jev-1.13.0`, `api.typesafe.ai/v1/systemone`).
The ext set is a first-pass queue from the largest dataset — its Jev
verdicts are 29% `content` (vs 0.2% on resplit) and 41% of its states
exceed 1024 tokens (`truncated_inputs` 5,743), so it also exercises the
long-input path.

### Task fidelity

The benchmark reuses the exact `state` string and `questions` payload of
`tools/judge_jev.py` (its sha256 is recorded in the report metadata): a
Japanese corpus-cleansing choice — is the marked suffix `noise`
(removable acquisition junk), `content` (story text to keep), or
`uncertain`. Jev's context window was `context[:6000]` chars; JevLite
encodes ≤1024 tokens, so long states lose tail characters Jev saw —
counted as `truncated_inputs` (154 of 11,842, 1.3%).

This is an exact-task reproduction on the input side; the only difference
is the engine behind `decide()`.

## Level 1 — paired agreement (NOT accuracy)

Jev is a reference system, not ground truth. These numbers measure how
much of Jev's historical *behavior* JevLite reproduces.

| backend | paired | agreement | disagreements |
|---|---|---|---|
| coreml `cpu_and_ne` (worker) | 11,842 | 0.17% | 11,822 |
| coreml `cpu_only` (worker) | 11,842 | 0.24% | 11,813 |
| torch `mps` | 11,842 | 0.15% | 11,824 |
| torch `cpu` | 11,842 | 0.17% | 11,824 |
| coreml `cpu_and_ne`, 1500m ext set | 14,046 | 0.10% | 14,032 |

The 1500m ext run reproduces the picture on a different distribution —
Jev said `content` for 4,109 items (29%) there; JevLite still chose
`uncertain` for 13,989 (99.6%), matching only 14 verdicts.

Verdict distributions (cpu_and_ne run; other backends within fp noise):

| | noise | content | uncertain |
|---|---|---|---|
| historical Jev | 11,814 | 28 | 0 |
| JevLite | 21 | 3 | 11,818 |

The four backends disagree with Jev in slightly different borderline
cases (fp16 vs fp32 numerics) but tell an identical story: JevLite, a
~64%-accurate model trained on agent-trace decisions — an entirely
different domain — answers `uncertain` at high confidence almost
everywhere, while Jev's surviving verdicts never use `uncertain`.

Of JevLite's 24 non-`uncertain` verdicts, 20 matched Jev's `noise` —
when the model does commit, it is mostly right, but that covers 0.2% of
the workload.

## Level 2 — human review sample

`artifacts/jev_human_review.jsonl` — 400 items, seed 42, stratified over
agreement × confidence × length. `human_label` is blank; fill it in and
rerun with `--gold artifacts/jev_human_review.jsonl --items-in
artifacts/jev_items_coreml_cpu_and_ne.jsonl` for per-system
accuracy/P/R/F1, confusion matrices, and Brier/ECE calibration.

**No human labels exist yet — every agreement number above is against
Jev, not truth.**

## Confidence routing

Would a hybrid router (high-confidence local → escalate the rest to Jev)
work? From the `cpu_and_ne` run:

| JevLite conf ≥ | handled locally | agreement w/ Jev | escalated |
|---|---|---|---|
| 0.60 | 97.7% | 0.0% | 2.3% |
| 0.70 | 93.1% | 0.0% | 6.9% |
| 0.80 | 81.0% | 0.0% | 19.0% |
| 0.90 | 46.9% | 0.0% | 53.1% |
| 0.95 | 15.6% | 0.0% | 84.4% |

Confidence does **not** stratify agreement: JevLite is *confidently*
`uncertain` where Jev never says `uncertain`. Raising the threshold only
shrinks coverage — it never finds a confident-agreement region. A hybrid
router gains nothing on this checkpoint. (Routing statistics only — no
superiority claim is possible without human gold.)

## Level 3 — systems benchmark

11,842 items, batch=1, input order preserved, timing includes
tokenization, first 50 items excluded from percentiles. Apple M2 Max,
macOS 26.6.2, PyTorch 2.14, coremltools 9.0.

| backend | items | load | p50 | p95 | p99 | dec/s | tok/s | restarts | recycles | parent RSS Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| coreml `cpu_and_ne` (worker) | 11,842 | 43.0 s | 113 ms | 117 ms | 139 ms | 8.5 | 6,487 | 0 | 1 | −470 MB |
| coreml `cpu_only` (worker) | 11,842 | 13.4 s | 194 ms | 223 ms | 275 ms | 5.0 | 3,830 | 0 | 1 | −473 MB |
| torch `mps` | 11,842 | 9.1 s | 54 ms | 68 ms | 78 ms | 18.2 | 13,900 | — | — | −474 MB |
| torch `cpu` | 11,842 | 7.8 s | 217 ms | 294 ms | 382 ms | 4.4 | 3,367 | — | — | −738 MB |
| coreml `cpu_and_ne`, ext set | 14,046 | 33.0 s | 112 ms | 116 ms | 131 ms | 8.6 | 6,610 | 0 | 1 | −459 MB |

Reading them:

- **`cpu_and_ne` beats `cpu_only` 1.7×** (113 vs 194 ms p50) on the real
  workload — the ANE path is doing real work, GPU excluded. torch `mps`
  still wins outright (54 ms); ANE's edge is GPU-free efficiency.
- Worker `load` (43 s) covers spawn + 3 package loads incl. ANE compile —
  reported separately from inference, as required.
- The single `recycle` per Core ML run is the planned `recycle_every=
  10,000` respawn that bounds `_keepalive`; its ~42 s reload appears in
  `max` latency (41.9 s) while p99 stays at 139 ms.
- **Zero crashes across ~37.7k Core ML worker predicts** — the
  probabilistic ANE heap bug did not fire inside a worker in any run;
  if it had, the parent would have survived it.
- Negative parent-RSS deltas are allocator noise — no growth trend over
  11.8k proxied requests. Worker RSS held ≈400 MB post-recycle.
- Historical Jev latency is **not comparable** and was never recorded in
  the verdict artifacts — hosted API, unknown wall-clock. Nothing
  invented.

## Limitations

- **JevLite checkpoint is the wrong tool for this task** — trained on
  ~1k agent-trace decisions, not Japanese corpus cleaning. Agreement
  measures the checkpoint, not the harness.
- Agreement-with-Jev is not accuracy. Human gold doesn't exist yet; the
  stratified sample is ready to label.
- 154/11,842 inputs (1.3%) exceed 1024 tokens — Jev saw their full
  6000-char context, JevLite saw less.
- Resplit verdicts are a filtered second pass: `uncertain` never
  survives into them, bounding achievable agreement by construction.
- No batching (batch=1), single machine.

## Reproduce

```bash
J=/path/to/j72-350m-mac   # novllm build tree - data stays outside this repo
python benchmark_jev_competition.py \
  --input $J/judge/longspan/queue_*.resplit.jsonl \
  --jev-results $J/judge/longspan/verdicts_*.resplit.jev.jsonl \
  --backend coreml:cpu_and_ne --out artifacts/jev_competition.json
# backends: coreml:cpu_and_ne | coreml:cpu_only | torch:mps | torch:cpu
# re-analyse without inference:
#   --items-in artifacts/jev_items_coreml_cpu_and_ne.jsonl
# human-gold scoring once labels exist:
#   --gold artifacts/jev_human_review.jsonl (same --items-in)
```

`--live-jev` hits the real API (`TYPESAFE_API_KEY`) — opt-in only, never
default, never bulk.
