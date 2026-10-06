# Diagnostic retrain — serialization + per-epoch label permutation

Follow-up to `SERIALIZATION_FIX.md`. The fix is now wired into the canonical
trainer: `02_train.py --slots {none,lead,pair}` + `--permute/--no-permute`
(default on). `tr_ds.resample(seed + ep)` re-encodes train rows each epoch;
validation always encodes in fixed order under the same `slots`. The winning
serialization is stored in the checkpoint (`ck["slots"]`) and every eval path
(05_serve / engine / coreml_backend+worker / 03 / 04 / verify_coreml /
judgebench) reads it back, so a checkpoint is always scored under the
serialization it was trained on.

## Conditions (identical except serialization × permutation)

train split `all` → 1016 train / 184 val (hash split, same rows every run),
`--seed 0`, 6 epochs fixed (patience 0), bs 4 × accum 4, lr 3e-5 enc / 1e-3
head, cosine, max_len 1024, ModernBERT-base, MPS.

| condition | flags | best val acc | ckpt |
|---|---|---|---|
| flat / no permutation (pre-fix regime) | `--slots none --no-permute` | 0.5946 (ep6) | `artifacts/diag_flat.pt` |
| lead / permutation (new default) | `--slots lead --permute` | 0.5478 (ep5) | `artifacts/diag_lead.pt` |
| pair / permutation | not yet run | – | – |

## pos0 selection rate on the synthetic fixture (`posprior_probe --eval`, own ser)

| k | diag_flat | diag_lead | uniform |
|---|---|---|---|
| 2 | 0.025 | 0.762 | 0.50 |
| 3 | 0.000 | 0.975 | 0.33 |
| 4 | 0.000 | 0.887 | 0.25 |
| 5 | 0.000 | 0.608 | 0.20 |
| 6 | 0.000 | 0.296 | 0.17 |

pos0 suppression is gone under lead+permute — overshooting toward pos0 at
k≤5 rather than never picking it. Accuracy on the fixture stays ≈1/k in
both conditions; permutation consistency ≤0.02 (binding still does not
transfer at this scale).

## JudgeBench, same backend for every row (torch/MPS, head-150 subset)

`--limit 150` on `artifacts/judgebench_620.jsonl`; subset pair_ids + sha256
recorded in each summary's `meta.subset`. Raw rows gitignored.

| system | acc (abstain=wrong) | coverage | acc\|decided | abstain | strict-pair | swap-consistency | first-pos share |
|---|---|---|---|---|---|---|---|
| jevlite.pt @ flat (before, native) | 0.117 | 0.223 | 0.522 | 0.777 | 0/150 | 0.640 | 0.00 |
| jevlite.pt @ lead (ser fix only, earlier run) | 0.010 | 0.027 | 0.375 | 0.973 | 0/150 | 0.947 | 0.75 |
| diag_flat (flat, no perm, 6ep) | 0.320 | 0.627 | 0.511 | 0.373 | 0/150 | 0.160 | 0.00 |
| diag_lead (lead, perm, 6ep) | 0.063 | 0.133 | 0.475 | 0.867 | 0/150 | 0.787 | 0.95 |

## Interim verdict (2/3 conditions complete)

- **Position usage is no longer structurally locked.** Fresh flat training
  still never picks pos0 (first-pos 0.00) *and* is wildly inconsistent under
  swap (0.16). lead+permute restores first-position usage (0.95 of decided
  calls) and swap consistency (0.79) — the direction inverted rather than
  vanishing.
- **Accuracy/coverage did not improve** — diag_lead decides less and scores
  lower than diag_flat on this OOD suite; strict-pair stays 0/150 for every
  condition. 6-epoch diagnostic budget is short of the full run; the
  incomplete pair condition may still add signal.

Per the task's own 判定 rule this is *not yet* "sufficiently improved" —
if the pair condition and/or a longer run do not move strict-pair accuracy,
the next investigation target is encoder-internal positional bias
(RoPE/position representation), as planned.

## Commands

```bash
python 02_train.py --slots none --no-permute --epochs 6 --patience 0 --seed 0 --out artifacts/diag_flat.pt
python 02_train.py --slots lead --permute    --epochs 6 --patience 0 --seed 0 --out artifacts/diag_lead.pt
python 02_train.py --slots pair --permute    --epochs 6 --patience 0 --seed 0 --out artifacts/diag_pair.pt

python bench/posprior_probe.py --eval artifacts/diag_flat.pt --ser flat
python bench/posprior_probe.py --eval artifacts/diag_lead.pt --ser sym
python bench/posprior_probe.py --eval artifacts/diag_pair.pt --ser pair

python bench/judgebench_eval.py --system jevlite-torch-baseflat --ckpt jevlite.pt --slots none --limit 150 \
    --out artifacts/judgebench_raw_diag_baseflat.jsonl --summary artifacts/judgebench_summary_diag_baseflat.json
python bench/judgebench_eval.py --system jevlite-torch-diagflat --ckpt artifacts/diag_flat.pt --limit 150 \
    --out artifacts/judgebench_raw_diag_flat.jsonl --summary artifacts/judgebench_summary_diag_flat.json
python bench/judgebench_eval.py --system jevlite-torch-diaglead --ckpt artifacts/diag_lead.pt --limit 150 \
    --out artifacts/judgebench_raw_diag_lead.jsonl --summary artifacts/judgebench_summary_diag_lead.json
```

Artifacts: `artifacts/judgebench_summary_diag_*.json` (committed),
`artifacts/posprior_metrics.json` (`diag_flat.pt`, `diag_lead.pt` keys),
`artifacts/diag_*.pt` (gitignored, ~600 MB each).
