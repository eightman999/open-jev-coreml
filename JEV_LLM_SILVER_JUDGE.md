# Jev vs JevLite — LLM Silver-Judge Evaluation

Independent third-party LLM judgement of the 400-case human-review sample
produced by `benchmark_jev_competition.py`. Two anonymized candidate decisions
(historical Jev and JevLite) were compared per case, twice, with candidate
order swapped between passes.

> **These are LLM-generated silver labels, not human gold labels.**
> Nothing here is ground truth; silver agreement is not accuracy.

## Dataset

- Source: `artifacts/jev_human_review.jsonl` — 400 cases, seed 42, unchanged
- Source SHA-256: `bf0acddabc0fbb5d72ef187d540136f32e3cee4a3dbe3ad2386716ce4d7e501a`
- Full state/inputs rejoined from the original resplit queues
  (SHA-256s listed in `JEV_COMPETITION.md`); the review file's `context`
  field (~300 chars) was **not** used — judges saw the full `jev_state`
  (≤6000 chars) the models were evaluated on
- Prompt template: `bench/prompts/llm_silver_judge_v1.txt`
  SHA-256 `bca4cde3e24461068fba7457c4aa3166c82129e2f2afc82105fc7950f6ae0f9d`

## Judges

Primary judge (all 400 cases, 2 passes):

```json
{
  "model": "gemma-4-31b-it",
  "backend": "hosted API (OpenAI-compatible)",
  "quantization": "not disclosed",
  "context_length": 1048576,
  "temperature": 0,
  "host": "hai-api.hcloud.ltd"
}
```

Second judge (partial, 16 complete case-pairs — lane stopped early by user
decision):

```json
{
  "model": "Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated (GGUF)",
  "quantization": "Q4_K_M",
  "backend": "llama.cpp 0.4.1-dev (b29c606)",
  "context_length": 8192,
  "temperature": 0,
  "host": "llmmaster (<private-host>, RTX 3060 + CPU expert offload)"
}
```

Candidate anonymization: prompts contained only `CANDIDATE A`/`CANDIDATE B`;
source identity (Jev/JevLite), agreement stats and confidence were never
disclosed. Base order per case and the pass-2 swap are deterministic
(SHA-256 of `queue_id`/`queue_id:pass2` + salt `silver-judge-v1`).

## Reliability

| metric | value |
|---|---|
| calls completed | 800 / 800 (400 cases × 2 passes) |
| swap consistency | **399 / 400** (99.75%) |
| refusals | 0 |
| timeouts / HTTP errors (final) | 0 |
| strict-JSON failures | 102 (12.8%), all recovered by field salvage |
| position A rate (decided calls) | 50.1% — no position bias |
| mean judge confidence | 1.00 (see limitations) |
| second-judge agreement | **0 / 16** — see below |

Note on strict-JSON failures: gemma frequently embeds unescaped double quotes
inside the `reason` string (e.g. `classified as "noise"`). A regex salvage
path in `parse_verdict` recovered all 102; raw responses are preserved in
`jev_llm_judge_raw.jsonl` for audit.

## Results (primary judge: gemma-4-31b-it)

Normalized silver verdicts over 400 cases:

| verdict | count |
|---|---|
| JEV_ONLY (historical Jev alone justified) | **399** |
| JEVLITE_ONLY | 0 |
| BOTH | 0 |
| NEITHER | 0 |
| AMBIGUOUS | 0 |
| UNRELIABLE / inconsistent → review | 1 |

Acceptability (stable silvers, n=399):

| candidate | acceptable rate |
|---|---|
| Jev | **1.00** |
| JevLite | **0.00** |

### JevLite `uncertain` analysis

All 400 sampled cases had `jevlite_verdict == "uncertain"` (the sample was
drawn from disagreement cases, where JevLite almost always said uncertain).

- uncertain judged **acceptable**: 0
- uncertain judged **unacceptable**: 399
- uncertain ambiguous: 0
- historical Jev's choice on the same cases: `noise` 399, `content` 1

Under this judge, JevLite's ~99.8% `uncertain` output is **degeneration,
not justified conservatism**: the candidate spans (author appeals, update
notices, site navigation) were judged clearly classifiable, and abstaining
was scored inappropriate. Notably, JevLite's answer was never judged the
uniquely correct one — by either judge.

## Second-judge check (n=16)

The qwen3-30b lane was stopped early (local GPU resource); its 16 complete
case-pairs were compared against the primary silvers:

- agreement: **0 / 16**
- qwen verdict on all 16: `BOTH` (both candidates acceptable)
- gemma verdict on the same 16: `JEV_ONLY`

The two judges diverge systematically on how to treat `uncertain`:
gemma penalizes abstention on classifiable text; qwen treats abstention as
defensible. **This is a judge-disposition difference, not ground truth** —
but even the lenient judge never preferred JevLite exclusively.

## Truncation analysis (JevLite input tokens)

| bucket | n | jev acceptable | jevlite acceptable | judge ambiguous | mean conf |
|---|---:|---:|---:|---:|---:|
| ≤1024 | 397 | 1.00 | 0.00 | 0.00 | 1.00 |
| >1024 (truncated) | 3 | 1.00 | 0.00 | 0.00 | 1.00 |

The review sample is nearly all ≤1024 tokens (resplit queues contain short
states); the >1024 slice is too thin (n=3) to support conclusions. The
fine buckets 0-512 / 1537+ are empty; 513-1024 ≈ ≤1024.

## Human-review reduction

- 400 original cases → **399 stable silver labels** → **1 case** requires
  human review (`artifacts/jev_llm_review_required.jsonl`)
- The single review case (`a2e0191e1c864bd9`): judge picked position A in
  both passes (position-inconsistent) on a `content` vs `uncertain`
  disagreement — a genuinely borderline candidate.

## Judge performance

| metric | value |
|---|---|
| total calls | 800 |
| mean latency | 7.1 s |
| p50 / p95 latency | 7.1 s / 10.2 s |
| sum latency | 5,713 s (~1.6 h compute, ~35 min wall at concurrency 4) |
| prompt tokens | 643,718 |
| completion tokens | 96,681 |
| endpoint failures in final data | 0 |

## Incidents during the run (recorded for reproducibility)

- llama.cpp V100 (sm_70) offload crashed repeatedly with
  `CUDA error: invalid argument` in `mul_mat_vec_q` — Qwen lane fell back to
  the proven RTX 3060 + CPU-expert config; V100 was excluded.
- Early gemma rows stored `PARSE_FAILURE`/`REFUSAL` verdicts caused by a
  max_tokens=512 truncation and an over-broad refusal regex matching quoted
  candidate text. Both bugs fixed; final analysis re-classifies from
  `raw_response`, so all 800 rows reflect the corrected classifier.
- Mid-run endpoint outages produced transient `HTTP_ERROR`/`TIMEOUT` rows;
  retryable rows were re-judged and superseded by later rows
  (last-wins per `(queue_id, pass)`).

## Artifacts

| file | content |
|---|---|
| `artifacts/jev_llm_judge_raw.jsonl` | all 800 calls: prompts hash, raw responses, verdicts, latency, tokens, judge identity (gitignored — embeds full text) |
| `artifacts/jev_llm_judge_raw.qwen30b.jsonl` | partial second-judge rows (gitignored) |
| `artifacts/jev_llm_silver_labels.jsonl` | 400 normalized silver labels + per-pass verdicts + review flags |
| `artifacts/jev_llm_review_required.jsonl` | 1 case for human review |
| `artifacts/jev_llm_judge_summary.json` | full metrics + reproducibility metadata |
| `bench/prompts/llm_silver_judge_v1.txt` | judge prompt template |
| `bench/llm_silver_judge.py` | harness (OpenAI-compatible, resumable, shardable) |

## Limitations

- **These are LLM-generated silver labels, not human gold labels.**
- Silver agreement is not accuracy; the 1.00/0.00 acceptability split is
  one judge's disposition applied consistently.
- **Judge bias**: gemma-4-31b-it penalizes abstention strongly; the partial
  qwen-30b judge treated `uncertain` as defensible in all 16 shared cases
  (0/16 agreement). Judge choice materially changes the verdict on
  abstention-heavy cases — the dominant direction (Jev's answers always
  acceptable, JevLite's never uniquely preferred) is consistent across both.
- **Common-model-family bias**: judges are general LLMs evaluating a
  narrow fine-tuned classifier on an out-of-distribution task; shared
  pretraining priors may correlate with either candidate's style.
- **Prompt sensitivity**: verdicts are conditioned on this prompt; the
  instruction "do not prefer a candidate because it is more cautious" may
  itself bias against abstention.
- **Position bias**: measured near-zero (A 50.1% / B 49.9%, 399/400
  swap-consistent) for gemma on this task — not guaranteed for other judges.
- **Truncation**: only 3/400 cases exceeded 1024 tokens, so truncation
  impact is not measurable on this sample.
- **Confidence calibration**: gemma emitted confidence 1.0 on every call;
  the confidence threshold therefore did no useful filtering — treat
  confidence as uninformative from this judge.
- Ambiguous cases: a judge that never says AMBIGUOUS may be forcing
  decisions on genuinely unclear text; the 1 flagged case is the floor of
  the ambiguity rate, not a measurement of it.

## Suggested follow-up diagnostics

Since the primary judge finds JevLite's `uncertain` collapse inappropriate,
worthwhile next steps (spec §20): per-class mean logit/probability and
prediction frequency on this workload; training label distribution and
class weights; choice-index/label-ID mapping audit; `uncertain` threshold
sensitivity; comparison against `judge_jev.py` choice semantics.

## Reproduction

```bash
# primary judge shard (HAI)
python bench/llm_silver_judge.py \
  --input <resplit queues> \
  --endpoint https://hai-api.hcloud.ltd/v1 --model gemma-4-31b-it \
  --shard 1/2 --output artifacts/jev_llm_judge_raw.gemma31b.jsonl \
  --passes 2 --temperature 0 --concurrency 4

# merge + analyze-only
cat artifacts/jev_llm_judge_raw.gemma31b*.jsonl \
  > artifacts/jev_llm_judge_raw.jsonl
python bench/llm_silver_judge.py --analyze-only \
  --input <resplit queues> \
  --output artifacts/jev_llm_judge_raw.jsonl \
  --judge2-file artifacts/jev_llm_judge_raw.qwen30b.jsonl \
  --endpoint x --model gemma-4-31b-it
```
