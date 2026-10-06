# JevLite never-A root-cause audit

Why did JevLite emit `candidate_a` **zero times in 1240 JudgeBench calls**?
Audit of `jevlite.pt` on `feat/judgebench-external-gold`. Artifacts:
`artifacts/jevlite_label_audit.json`, `jevlite_head_audit.json`,
`jevlite_synthetic_audit.jsonl`, `jevlite_permutation_audit.jsonl`.

## Architecture recap

There is **no class index and no per-class head**. `iter_labels` sorts the
criteria alphabetically; `td_data.encode` lays one `<<l>>` marker per label in
that order; a single shared `Linear(d,1)` scores every marker position; CPU
gathers `label_pos` and applies a grouped softmax. "Class index" in this audit
therefore means *position within the sorted criteria list*.

## Verdict

**CONFIRMED: the suppression tracks the first sorted label position, not the
`candidate_a` meaning — a learned positional prior that collapses on
out-of-schema (3-label) inputs.**

## Evidence

### D5 — permutation test (decisive)

Four layouts put the same three meanings (A-wins / B-wins / uncertain) at
different sorted positions (label names `aa`/`bb`/`cc`, identical
descriptions). 48 calls, 6 obvious-A synthetic pairs × 2 orders × 4 layouts:

| layout | pos0 wins | pos1 wins | pos2 wins | pos0 mean prob |
|---|---|---|---|---|
| A-B-U | 0 | 8 | 4 | 0.130 |
| B-A-U | 0 | 6 | 6 | 0.148 |
| U-A-B | 0 | 3 | 9 | 0.114 |
| A-U-B | 0 | 0 | 12 | 0.090 |

Position 0 wins **0/48** regardless of which meaning sits there. The
pathology follows the position index — not the label name and not the
"candidate A wins" semantics.

### D4+D7 — synthetic suite (replication, not truncation)

20 short cases × 2 orders, all ≪512 tokens: torch verdicts `uncertain` 39 /
`candidate_b` 1 / `candidate_a` 0; CoreML identical (39/1/0). Even
obvious-A/obvious-B fixtures abstain. Pathology reproduces on short inputs →
**not attributable to context truncation**.

### D3 — head + logit stats

Shared head `Linear(768,1)`: bias −0.019, weight L2 0.40, mean 5.8e-5,
std 0.014 — unremarkable, and per-class head collapse is structurally
impossible with one shared head. Per-label logits over 60 JudgeBench cases:

| label | mean | median | max |
|---|---|---|---|
| candidate_a (pos0) | −0.85 | −0.76 | **−0.08** |
| candidate_b (pos1) | +0.49 | +0.59 | +1.27 |
| uncertain (pos2) | +0.79 | +0.78 | +1.75 |

pos0 logits never even reach zero.

### D1 — training label distribution (`LocalLLaMA/typed-decisions`)

- train 1200 rows / test 400 rows; no validation split.
- **choice questions exist only at 4 and 5 labels** (1200 + 600 train
  question instances): zero 3-label choice questions — the JudgeBench schema
  size was never trained.
- `uncertain`, `candidate_a`, `candidate_b` never appear as label names
  (training labels are `true/false`, `0..4`, `continue/stop/...` etc.).
- Position argmax in training (choice): pos0 349, pos1 364, pos2 809,
  pos3 155, pos4 123 — position 2 dominant, position 3 suppressed, but
  **position 0 was NOT dead** (~19% of choice argmaxes). The dead-position-0
  is not copied from training targets.

### D2 — label ↔ position mapping

`iter_labels → encode.label_pos → gather → softmax → decide key` verified:
`enc["label_pos"]` points exactly at `<<l>>` marker ids (checked by decoding
`input_ids[p:p+4]`), label order `candidate_a, candidate_b, uncertain` is
consistent end-to-end by construction (single shared `iter_labels` code path
for training, torch inference, CoreML export input, and worker IPC).

### D6 — baseline sanity

Metric denominators verified by unit tests (always-A / always-B /
always-uncertain / alternating / pseudo-random judges produce the expected
overall/coverage/strict/consistency values) — the measurement itself is not
the artifact source.

## Cause classification

| hypothesis | verdict |
|---|---|
| CoreML conversion bug | **RULED_OUT** — Torch reproduces verdicts identically (synthetic 39/1/0 on both) |
| worker/IPC bug | **RULED_OUT** — same reproduction on in-process Torch |
| enum/label mapping bug | **RULED_OUT** — single `iter_labels` path; decoded positions verified |
| encode off-by-one | **RULED_OUT** — `label_pos` sits on `<<l>>` exactly |
| per-class head collapse | **RULED_OUT** — one shared Linear(d,1) scores all positions; no per-class parameters exist |
| truncation | **RULED_OUT** — short synthetic inputs (<512 tok) reproduce |
| label-name semantics (`candidate_a` unreadable) | **RULED_OUT** — suppression follows position across permuted names |
| training label imbalance | **STRONGLY_SUPPORTED as contributor, RULED_OUT as direct mechanism** — pos0 wasn't dead in training, but the 4-5-label-only distribution plausibly shaped the positional prior |
| first-position positional prior | **CONFIRMED** — pos0 wins 0/48 under meaning permutation |
| input representation / schema OOD (3-label) | **CONFIRMED contributor** — zero 3-label choice questions in training; novel label count + novel label names simultaneously |
| exact encoder-internal mechanism | **POSSIBLE** — why pos0 specifically (instruction-adjacency, attention dynamics) is not dissected |

## Recommendation

No code fix exists — the positional prior lives in the trained weights.
`jevlite.pt` should **not** be used for dynamic-schema judging as-is.

Before committing to a full retrain, run the cheap diagnostic the spec
allows: fine-tune on a small (hundreds–thousands) synthetic set that varies
(a) label count 2–6, (b) label names (random strings, not just semantic
words), (c) gold position — never JudgeBench rows. Success criterion:
`candidate_a`-position wins on the synthetic fixture and pos0 mean logit
normalizes. If a few-hundred-example diagnostic restores position-0
predictions, the checkpoint is repairable; if not, retrain from scratch with
label-count and label-name diversity plus position-shuffled gold.

## JudgeBench metric/naming fixes applied alongside this audit

- Outcome matrix previously computed on pass-1 only but labeled "merged by
  strict rule" — now emitted as `outcome_matrix_original`,
  `outcome_matrix_swapped`, `outcome_matrix_strict_pair` (+ combo detail).
- `accuracy` → `overall_accuracy_abstain_wrong`, `conditional_accuracy` →
  `accuracy_given_decision`, truncation `forced_acc` →
  `accuracy_given_decision`, new `binary_forced_argmax_accuracy`
  (Jev 0.752 / JevLite 0.491).
