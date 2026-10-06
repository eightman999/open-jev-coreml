# pos0 Serialization Fix — lead/pair dummy slots + train-time permutation

Follow-up to `JEVLITE_NEVER_A_AUDIT.md` / `BASE_MODEL_POSITION_AUDIT.md`.

## Diagnosis that motivated this

The first `<<l>>` marker of every question sits directly after the instruction
text, while every later `<<l>>` sits after a label-shaped `"name: desc"` block.
That single structural asymmetry is what lets position 0 be singled out: an
*untrained* encoder + random head already picks pos0 in ~0/1200 synthetic
cases, and the trained checkpoint inherits the suppression out-of-distribution.

## Fix implemented

`td_data.encode(row, tok, max_len, qid, lid, slots=..., rng=...)`:

- `slots="lead"` (**new default**): one fixed, never-gathered
  `<<l>> "-: options"` block before each question's first real label.
- `slots="pair"`: one dummy block before *every* label plus one trailing —
  every gathered marker has the identical `<<l>> SLOT_TEXT` predecessor.
- `slots="none"`: the original flat layout (kept for before/after runs).
- `rng=random.Random(...)`: shuffles label emission order per question
  (all types except `score`, whose order is semantic). Gold follows by label
  name — training augmentation only, never used at inference.
- `TypedDecisions(..., permute=True)` + `ds.resample(seed)` re-encodes all rows
  per epoch with a fresh permutation.

Cost: `lead` adds ~4 tokens per question (≤ ~32 tokens worst case at MAX_Q=8);
`pair` adds ~5 per label. The state-truncation policy is unchanged — the extra
slots draw from the same `max_len` budget.

## Init-time position bias (base encoder + random head, no training)

pos0 selection rate, 240 cases per label-count k:

| k | flat | lead | pair (+trail) | uniform |
|---|---|---|---|---|
| 2 | 0.00 | 0.15 | 0.75 | 0.50 |
| 3 | 0.00 | 0.71 | 0.75 | 0.33 |
| 4 | 0.00 | 0.80 | 0.83 | 0.25 |
| 5 | 0.00 | 0.94 | 0.43 | 0.20 |
| 6 | 0.00 | 0.82 | 0.30 | 0.17 |

**Serialization cannot fully flatten init bias** — under random weights the
encoder's chain-position dynamics always favor some position. Flat is uniquely
bad because it *always* suppresses pos0; slot variants move the bias rather
than remove it. The residual is encoder-level (RoPE chain position), matching
the `BASE_MODEL_CAUSE_SUPPORTED` verdict.

## Frozen / partial / full minimal comparison (lead ser + permutation, 800 rows)

| condition | pos0 sel (k=2..6) | learned? |
|---|---|---|
| frozen (head+marker emb, 3ep) | 1.0/.98/.56/.38/.20 | no (loss flat) |
| partial4 (last 4 layers, 5ep) | .59/.25/.53/.58/.53 | barely (loss ~2.0) |
| full (8ep, enc-lr 1e-4) | .50/.63/.78/.73/.71 | no — permutation blocks memorization, eval stays ≈ chance |

Small-scale synthetic training does not converge on this binding fixture
(consistent with the prior audit's finding that content→position binding
doesn't transfer at 800-example scale). Init bias therefore persists in these
probes — this bounds what the synthetic suite can prove, not the fix's value.

## Key result: the existing checkpoint under the new serialization

`jevlite.pt` (unchanged weights) on the synthetic position fixture:

| k | flat pos0 | lead pos0 | pair pos0 | uniform |
|---|---|---|---|---|
| 2 | 0.00 | 0.41 | 0.22 | 0.50 |
| 3 | 0.00 | 0.30 | 0.30 | 0.33 |
| 4 | 0.00 | 0.17 | 0.30 | 0.25 |
| 5 | 0.00 | 0.28 | 0.14 | 0.20 |
| 6 | 0.03 | 0.34 | 0.05 | 0.17 |

**pos0 suppression is a context artifact, not a weight property.** Giving the
first marker a label-shaped left neighbour restores near-uniform pos0 usage in
the existing checkpoint — no retraining required for the mechanism itself.

## JudgeBench before/after (identical 150-pair subset, 300 calls each)

| metric | flat (CoreML worker) | lead (torch/MPS) |
|---|---|---|
| candidate_a wins | **0/300** (maxP 0.27) | **6/300** (meanP 0.22 ≈ 1/3) |
| candidate_b wins | 71 | 2 |
| uncertain | 229 | 292 |
| coverage | 0.237 | 0.027 |
| abstention | 0.763 | 0.973 |
| overall_accuracy_abstain_wrong | 0.103 | 0.010 |
| accuracy_given_decision | 0.437 | 0.375 |
| strict-pair accuracy | 0/150 | 0/150 |
| first-position rate (decided calls) | 0.00 | 0.75 |

candidate_a is mechanically alive again (P(candidate_a) mean 0.11→0.22,
first-position share of decided calls 0%→75%), **but** the checkpoint's OOD
policy still collapses real inputs onto `uncertain` — now the last/ favoured
position absorbs even more mass. Strict-pair stays 0.

## Verdict

- **Serialization fix is necessary and works mechanically** — pos0/candidate_a
  is no longer structurally suppressed, in both the checkpoint and at init
  the unique "first marker after prose" handicap is gone.
- **Serialization alone is not sufficient** — the checkpoint's trained policy
  (prefer late positions / abstain) still fails as a judge on OOD schemas.
- **Encoder change is not the next step**: position usage is already
  near-uniform for the trained checkpoint under `lead`. The next step is the
  authorized *diagnostic retrain*: dynamic 2–6-label data, randomized label
  names, per-epoch permutation, `slots="lead"`, unfrozen encoder — then
  re-measure pos0 rates, JudgeBench strict-pair and abstention.
- `pair` mode exists but doubles marker count with no post-training benefit
  shown here; `lead` is the committed default.

## Commands

```bash
# init-bias probes (each serialization)
python bench/posprior_probe.py --eval UNTRAINED --ser flat|sym|pair
# minimal training comparison (800 synthetic rows, permutation on)
python bench/posprior_probe.py --train-a --mode frozen  --ser sym --epochs 3
python bench/posprior_probe.py --train-a --mode partial --unfreeze-last 4 --ser sym --epochs 5 --enc-lr 1e-4
python bench/posprior_probe.py --train-a --mode full    --ser sym --epochs 8  --enc-lr 1e-4
# checkpoint under fixed serialization
python bench/posprior_probe.py --eval jevlite.pt --ser sym|pair
# JudgeBench subset under new serialization (torch/MPS)
python bench/judgebench_eval.py --system jevlite-torch --limit 150 \
    --out artifacts/judgebench_raw_sym.jsonl \
    --summary artifacts/judgebench_summary_sym_probe.json
```

Artifacts: `artifacts/posprior_metrics.json` (all conditions, committed),
`artifacts/judgebench_summary_sym_probe.json` (committed),
`artifacts/judgebench_raw_sym.jsonl` + `posprior_*_{lead,pair}.pt`
(regenerable, gitignored).
