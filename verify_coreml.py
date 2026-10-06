"""Parity harness: PyTorch reference vs Core ML at each compute unit.

Two levels of comparison on identical encoded rows:

  logit     token_logits [1,S] gathered at the <<l>> marker positions. PyTorch
            runs the exact function that was exported on the exact padded input
            the package sees, so this isolates conversion error.
  decision  the reference forward path (grouped softmax + per-type temperature,
            same as 05_serve) vs CoreMLEngine.decide_enc - label agreement and
            confidence MAE per question.

Rows are encoded at min(ckpt max_len, largest bucket), so when the 1024 package
is present the comparison is the full untruncated pipeline.

  python verify_coreml.py --ckpt jevlite.pt --limit 50
"""
import argparse
import json
import time

import numpy as np
import torch

from td_data import load_split, TypedDecisions, collate
from model import JevLite, require_ckpt, argmax_per_question
from coreml_backend import CoreMLEngine

COMPUTE_UNITS = ["cpu_only", "cpu_and_ne", "all"]


def ref_label_logits(model, enc, S, dev, pad_id):
    """token_logits on the same padded [1,S] input the package sees."""
    L = min(len(enc["input_ids"]), S)
    ids = torch.full((1, S), pad_id, dtype=torch.long)
    att = torch.zeros(1, S, dtype=torch.long)
    ids[0, :L] = torch.tensor(enc["input_ids"][:L])
    att[0, :L] = 1
    with torch.no_grad():
        tl = model.token_logits(ids.to(dev), att.to(dev))[0].float().cpu()
    return tl[torch.tensor(enc["label_pos"])].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--pkg-dir", default="artifacts/coreml")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--compute-units", nargs="+", default=COMPUTE_UNITS,
                   choices=COMPUTE_UNITS)
    p.add_argument("--out", default="artifacts/parity_report.json")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    dev = "cpu"          # the reference number must be device-stable
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    engines = {cu: CoreMLEngine(a.pkg_dir, ckpt=a.ckpt, compute_units=cu)
               for cu in a.compute_units}
    max_seq = max(e.max_seq for e in engines.values())

    # The reference always encodes at the checkpoint's own max_len; rows whose
    # encoding no installed bucket can hold are skipped (reported, not hidden).
    rows = load_split(a.split, limit=a.limit)
    ds = TypedDecisions(rows, model.tok, ck["max_len"], model.qid, model.lid,
                        slots=ck.get("slots", "lead"))

    # --- reference passes on the identical encoded inputs --------------------
    t0 = time.time()
    ref_lg, ref_dec, encs, skipped = [], [], [], 0
    for i in range(len(ds)):
        enc = ds[i]
        if len(enc["input_ids"]) > max_seq:
            skipped += 1
            continue
        encs.append(enc)
        S = engines[a.compute_units[0]]._bucket(len(enc["input_ids"]))
        ref_lg.append(ref_label_logits(model, enc, S, dev,
                                       model.tok.pad_token_id))
        b = collate([enc], model.tok.pad_token_id)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            _, probs = model(b, apply_temperature=True)
        pred, conf = argmax_per_question(probs, b)
        gold, _ = argmax_per_question(b["target"], b)
        ref_dec.append({"pred": pred[0].cpu(), "conf": conf[0].cpu(),
                        "gold": gold[0].cpu(), "qmask": b["qmask"][0].cpu()})
    print(f"reference pass: {len(encs)} rows in {time.time()-t0:.0f}s "
          f"(max_len {ck['max_len']}, max bucket {max_seq}, "
          f"{skipped} skipped as unencodable)")

    report = {"ckpt": a.ckpt, "split": a.split, "n_rows": len(encs),
              "skipped": skipped, "max_bucket": max_seq,
              "compute_units": {}}

    for cu, eng in engines.items():
        errs, conf_errs = [], []
        agree = hit = ref_hit = tot = 0
        t0 = time.time()
        for i, enc in enumerate(encs):
            lg = eng.label_logits(enc)
            errs.append(np.abs(lg - ref_lg[i]))
            got = eng.decide_enc(enc)
            rd = ref_dec[i]
            for gi, qname in enumerate(enc["qnames"]):
                if gi >= rd["pred"].numel() or not rd["qmask"][gi]:
                    continue
                labels = enc["labels"][gi]
                r_pred, r_conf = int(rd["pred"][gi]), float(rd["conf"][gi])
                r_gold = int(rd["gold"][gi])
                g_pred = labels.index(got[qname]["label"])
                agree += int(g_pred == r_pred)
                conf_errs.append(abs(got[qname]["confidence"] - r_conf))
                if r_gold >= 0:
                    hit += int(g_pred == r_gold)
                    ref_hit += int(r_pred == r_gold)
                tot += 1
        errs = np.concatenate(errs)
        rep = {
            "max_abs_logit_err": float(errs.max()),
            "mean_abs_logit_err": float(errs.mean()),
            "decision_agreement": agree / max(tot, 1),
            "confidence_mae": float(np.mean(conf_errs)),
            "n_decisions": tot,
            "coreml_accuracy": hit / max(tot, 1),
            "torch_accuracy": ref_hit / max(tot, 1),
            "accuracy_regression_pp": (ref_hit - hit) / max(tot, 1) * 100,
            "wall_s": time.time() - t0,
        }
        report["compute_units"][cu] = rep
        print(f"{cu:>10s}  max|dlogit| {rep['max_abs_logit_err']:.4f}  "
              f"mean {rep['mean_abs_logit_err']:.5f}  "
              f"agree {rep['decision_agreement']:.4f}  "
              f"confMAE {rep['confidence_mae']:.4f}  "
              f"acc {rep['coreml_accuracy']:.4f} vs {rep['torch_accuracy']:.4f}  "
              f"({rep['wall_s']:.0f}s)")

    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
