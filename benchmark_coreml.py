"""Latency benchmark: PyTorch CPU/MPS vs Core ML CPU_ONLY/CPU_AND_NE/ALL.

The measured unit is one token_logits forward pass at a fixed [1, S] input -
the same computation every backend performs. A real encoded row is reused as
input so the numbers reflect production-shaped data, not randint noise.

  python benchmark_coreml.py --ckpt jevlite.pt --iters 100

Warmup is always excluded. Median/p90/p95 are reported, never best-case.
CPU_AND_NE is the ANE candidate: if it does not beat CPU_ONLY, say so - do not
report ALL (which may use the GPU) as "ANE speed".
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from td_data import load_split, TypedDecisions
from model import JevLite, require_ckpt

WARMUP = 20


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "p90_ms": pct(ts, 90),
            "p95_ms": pct(ts, 95), "mean_ms": float(ts.mean()),
            "min_ms": float(ts.min()), "max_ms": float(ts.max()),
            "req_per_s": 1000.0 / float(np.median(ts))}


def make_inputs(model, ckpt, seq_len):
    """One real encoded row padded to seq_len - identical for every backend."""
    rows = load_split("test", limit=8)
    ds = TypedDecisions(rows, model.tok, ckpt["max_len"], model.qid, model.lid)
    enc = max(ds.enc, key=lambda e: len(e["input_ids"]))   # longest of the 8
    L = min(len(enc["input_ids"]), seq_len)
    ids = np.full((1, seq_len), model.tok.pad_token_id, dtype=np.int32)
    att = np.zeros((1, seq_len), dtype=np.int32)
    ids[0, :L] = enc["input_ids"][:L]
    att[0, :L] = 1
    return ids, att, L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--pkg-dir", default="artifacts/coreml")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--backends", nargs="+",
                   default=["torch_cpu", "torch_mps",
                            "coreml_cpu_only", "coreml_cpu_and_ne",
                            "coreml_all"],
                   choices=["torch_cpu", "torch_mps", "coreml_cpu_only",
                            "coreml_cpu_and_ne", "coreml_all"])
    p.add_argument("--out", default="artifacts/benchmark.json")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    report = {"ckpt": a.ckpt, "iters": a.iters, "warmup": a.warmup,
              "results": {}, "load": {}, "package_mb": {}}

    model = None
    if any(b.startswith("torch") for b in a.backends):
        t0 = time.perf_counter()
        model = JevLite(ck["encoder"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        report["load"]["torch_cpu"] = time.perf_counter() - t0
        if "torch_mps" in a.backends and torch.backends.mps.is_available():
            t0 = time.perf_counter()
            mps_model = JevLite(ck["encoder"])
            mps_model.load_state_dict(ck["state_dict"])
            mps_model.to("mps").eval()
            report["load"]["torch_mps"] = time.perf_counter() - t0
        else:
            mps_model = None

    engines = {}
    if any(b.startswith("coreml") for b in a.backends):
        from coreml_backend import CoreMLEngine, _compute_unit
        import coremltools as ct
        for b in a.backends:
            if not b.startswith("coreml"):
                continue
            cu = b.replace("coreml_", "")
            t0 = time.perf_counter()
            engines[b] = {s: ct.models.MLModel(
                os.path.join(a.pkg_dir, f"jevlite-{s}.mlpackage"),
                compute_units=_compute_unit(cu))
                for s in a.seq_lens
                if os.path.exists(os.path.join(a.pkg_dir,
                                               f"jevlite-{s}.mlpackage"))}
            report["load"][b] = time.perf_counter() - t0

    for s in a.seq_lens:
        path = os.path.join(a.pkg_dir, f"jevlite-{s}.mlpackage")
        if os.path.exists(path):
            report["package_mb"][str(s)] = round(
                sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(path) for f in fs) / 1e6, 1)

    for s in a.seq_lens:
        ids, att, L = make_inputs(model, ck, s) if model is not None else \
            make_inputs(_TmpModel(ck), ck, s)
        report["results"][str(s)] = {}
        print(f"\n=== seq_len {s} (real tokens {L}) ===")
        for b in a.backends:
            if b == "torch_cpu":
                ids_t = torch.tensor(ids).long()
                att_t = torch.tensor(att).long()
                def fn():
                    with torch.inference_mode():
                        model.token_logits(ids_t, att_t)
            elif b == "torch_mps":
                if mps_model is None:
                    continue
                ids_t = torch.tensor(ids, device="mps").long()
                att_t = torch.tensor(att, device="mps").long()
                def fn():
                    with torch.inference_mode():
                        mps_model.token_logits(ids_t, att_t)
                    torch.mps.synchronize()
            else:
                if s not in engines[b]:
                    continue
                m = engines[b][s]
                feed = {"input_ids": ids, "attention_mask": att}
                fn = lambda: m.predict(feed)
            r = timeit(fn, a.iters, a.warmup)
            r["tokens_per_s"] = r["req_per_s"] * s
            report["results"][str(s)][b] = r
            print(f"  {b:>18s}  median {r['median_ms']:7.1f} ms  "
                  f"p90 {r['p90_ms']:7.1f}  p95 {r['p95_ms']:7.1f}  "
                  f"{r['req_per_s']:6.1f} req/s  {r['tokens_per_s']:8.0f} tok/s")

    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

    ne = report["results"].get(str(max(a.seq_lens)), {})
    if "coreml_cpu_and_ne" in ne and "coreml_cpu_only" in ne:
        speedup = (ne["coreml_cpu_only"]["median_ms"]
                   / ne["coreml_cpu_and_ne"]["median_ms"])
        print(f"CPU_AND_NE vs CPU_ONLY at seq {max(a.seq_lens)}: "
              f"{speedup:.2f}x {'(ANE helping)' if speedup > 1 else '(NO ANE WIN)'}")


class _TmpModel:
    """Tokenizer-only stand-in when only Core ML backends are benchmarked."""
    def __init__(self, ck):
        from transformers import AutoTokenizer
        from td_data import add_markers
        self.tok = AutoTokenizer.from_pretrained(ck["encoder"])
        _, self.qid, self.lid = add_markers(self.tok)


if __name__ == "__main__":
    main()
