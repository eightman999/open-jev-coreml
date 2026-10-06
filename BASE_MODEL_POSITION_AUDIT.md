# Base-Model Position Audit — is pos0 suppression inherent to the forked encoder?

**Verdict: `BASE_MODEL_CAUSE_SUPPORTED`** — position-0 suppression is present in the
*untrained* base encoder + serialization (random head, zero training steps). It is
not created by the JevLite fine-tune; the fine-tune merely failed to correct it
outside its training distribution. It is, however, **correctable**: unfrozen
fine-tuning on uniform-gold synthetic data removes the suppression, while a
frozen encoder cannot remove it through the head alone.

## Setup

- Probe: `bench/posprior_probe.py`; seed `20260922`.
- Data: `artifacts/posprior_train.jsonl` (800 rows), `artifacts/posprior_eval.jsonl`
  (1,200 rows = 600 cases × unrotated + rotated permutation pair).
- Sizes: 2–6 labels, 120 eval cases per size; gold position ~uniform per size;
  label names randomized per row (`gold`/`silver`/`bronze`/… drawn from a pool).
- Serialization mirrors the JudgeBench adapter: state carries `CANDIDATE A…F`
  blocks with random-word content; label descriptions carry the candidate word;
  exactly one designated-correct key per case. `MAX_LEN=256`, no truncation.
- Device: MPS. Metrics: `artifacts/posprior_metrics.json`.

## Conditions

| condition | encoder | head | training |
|---|---|---|---|
| `UNTRAINED` | base `answerdotai/ModernBERT-base`, untouched | fresh `Linear(d,1)`, seed init | none |
| `posprior_a.pt` (A-frozen) | frozen | fresh head + `<<q>>`/`<<l>>` embeddings trained | 3 ep — loss flat 2.11→2.09 (learned ~nothing) |
| `posprior_a_full.pt` (A-full) | unfrozen, lr 1e-4 | head lr 1e-3 | 15 ep, eff. batch 16 — train loss 2.06→0.22 (memorized); eval ≈ chance |
| `jevlite.pt` (B) | existing checkpoint | existing | the real fine-tune |

## Results

`pos0_selection_rate` (share of cases where the argmax landed on position 0)
and `accuracy` on the binding fixture:

| k | UNTRAINED pos0 | A-frozen pos0 | A-full pos0 | jevlite pos0 | uniform baseline |
|---|---|---|---|---|---|
| 2 | 0.000 | 0.200 | 0.358 | 0.004 | 0.50 |
| 3 | 0.000 | 0.000 | 0.392 | 0.000 | 0.33 |
| 4 | 0.000 | 0.000 | 0.296 | 0.004 | 0.25 |
| 5 | 0.004 | 0.000 | 0.246 | 0.000 | 0.20 |
| 6 | 0.004 | 0.000 | 0.229 | 0.033 | 0.17 |

Accuracy ≈ 1/k for every condition (all fail the binding task on random-word
content; permutation consistency ≤ 0.07 for all). The diagnostic signal here is
*position usage*, not accuracy.

`mean_logit_by_pos` for the **untrained** model — pos0 is the lowest at every
size (k=4 shown; others identical in shape):

```text
pos0: -0.74   pos1: -0.26   pos2: -0.11   pos3: -0.13
```

The first `<<l>>` marker's hidden state projects lowest under a random
direction, so an untrained head essentially never picks it — a monotone-ish
positional gradient favoring later markers.

## Interpretation

1. **pos0 suppression predates training.** A fresh encoder + random head selects
   position 0 in ~0/1,200 eval cases at every label count. The disadvantage
   lives in the encoder hidden states for this serialization, not in learned
   weights or the trained head.
2. **The prior is plastic, not architectural.** Unfrozen fine-tuning on
   uniform-gold synthetic data flipped the ordering (pos0 sel 0.23–0.39 ≈
   uniform; pos0 logit no longer lowest). A frozen encoder cannot do this —
   head + marker embeddings alone leave pos0 at 0 for k ≥ 3.
3. **JevLite's fine-tune did not robustly correct it.** The checkpoint keeps
   pos0 ≈ 0 on this OOD fixture and on JudgeBench. Its own training
   distribution (4/5-label data only, pos0 argmax ~19% in-distribution) masked
   the defect; the underlying encoder prior resurfaces out-of-distribution.
4. **Separate finding — binding does not transfer.** Neither the checkpoint nor
   the 800-example synthetic fine-tune learns content→position binding that
   survives rotation (perm. consistency ≤ 0.07). JevLite's real skill is
   distribution-bound; this does not change the pos0 conclusion but bounds
   what a small synthetic repair can prove.

## Repair implication

- A diagnostic repair fine-tune **must unfreeze the encoder** — a frozen-encoder
  head repair cannot recover position 0.
- Uniform gold positions across 2–6 labels are the right corrective signal:
  A-full shows the encoder can re-learn position usage.
- The binding-task failure at this scale means the synthetic suite should test
  *position coverage and schema-shape robustness*, not expect high accuracy on
  random-word content.

## Commands

```bash
python bench/posprior_probe.py --make-data
python bench/posprior_probe.py --train-a                                   # A-frozen, 3 ep
python bench/posprior_probe.py --train-a --unfreeze --epochs 15 --enc-lr 1e-4  # A-full
python bench/posprior_probe.py --eval UNTRAINED
python bench/posprior_probe.py --eval artifacts/posprior_a.pt
python bench/posprior_probe.py --eval artifacts/posprior_a_full.pt
python bench/posprior_probe.py --eval jevlite.pt
```

Artifacts: `artifacts/posprior_metrics.json` (committed),
`artifacts/posprior_{train,eval}.jsonl` (regenerable, gitignored),
`artifacts/posprior_a{,_full}.pt` (gitignored).
