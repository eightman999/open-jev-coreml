"""Step 5 - the Jev-shaped API, running locally for free.

  python 05_serve.py --demo
  python 05_serve.py --serve      # POST /decide {"state": {...}, "questions": {...}}

The questions are supplied per request. Nothing about the schema is baked into
the weights, so you can add a question, rename a label, or point it at a new
workflow without retraining - the same property that makes Jev usable as a
general decision endpoint.
"""
import argparse
import json
import torch
from td_data import encode, collate
from device import pick_device
from model import JevLite, require_ckpt
from typed_schema import iter_labels, Question

_engine = {}


def load(ckpt="jevlite.pt", device=None, backend="torch",
         compute_units="cpu_and_ne", pkg_dir="artifacts/coreml",
         coreml_mode="worker", slots=None):
    if backend == "coreml":
        # "worker" (default) keeps the ANE runtime in a subprocess - the heap
        # corruption in ANE_REPORT.md cannot take down this process.
        # "inprocess" is a debugging option only.
        from engine import _coreml_impl
        eng, suffix = _coreml_impl(pkg_dir, ckpt, compute_units, coreml_mode)
        _engine.update(model=eng, dev=f"coreml:{compute_units}{suffix}",
                       max_len=eng.encode_len, temps=eng.temperatures,
                       slots=eng.slots)
        return _engine
    ck = require_ckpt(ckpt)
    dev = device or pick_device()
    m = JevLite(ck["encoder"])
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    # A checkpoint trained under one serialization must be served under the
    # same one; checkpoints older than the slots field get the current
    # default ("lead"), which is the pos0 fix.
    _engine.update(model=m, dev=dev, max_len=ck["max_len"],
                   temps=ck.get("temperatures"),
                   slots=slots or ck.get("slots", "lead"))
    return _engine


@torch.no_grad()
def decide(state, questions, calibrated=True):
    """-> {question: {"label": str, "confidence": float, "probabilities": {...}}}"""
    e = _engine
    m = e["model"]
    if not isinstance(m, JevLite):
        # CoreMLEngine owns its own encode/pad/predict path
        return m.decide(state, questions, calibrated)
    row = {"id": "live", "state": state, "questions": questions, "gold": {}}
    enc = encode(row, m.tok, e["max_len"], m.qid, m.lid, with_gold=False,
                 slots=e.get("slots", "lead"))
    b = collate([enc], m.tok.pad_token_id)
    b = {k: v.to(e["dev"]) for k, v in b.items()}
    _, probs = m(b, apply_temperature=calibrated)
    probs = probs[0].cpu()

    out, i = {}, 0
    for gi, qname in enumerate(enc["qnames"]):
        labels = enc["labels"][gi]
        vals = probs[i:i + len(labels)].tolist()
        i += len(labels)
        dist = dict(zip(labels, (round(v, 4) for v in vals)))
        top = max(dist, key=dist.get)
        out[qname] = {"label": top, "confidence": dist[top], "probabilities": dist}
    return out


DEMO_STATE = {
    "agent": {"autonomy": "checkpointed", "model": "internal-agent-v4"},
    "constraints": ["Do not touch customer data outside the named accounts"],
    "task": "Delete personal data for the accounts in the erasure queue.",
    "trace_summary": {"duration_s": 128.5, "steps": 7, "tool_errors": 1,
                      "irreversible_actions": 2, "constraint_violations": 1},
}

DEMO_QUESTIONS = {
    "action": Question.choice(
        "What should the observability system do with this trace?",
        {"continue": "Let the agent proceed without interruption.",
         "human_review": "Queue this trace for a human to review.",
         "observe": "Keep running, but flag the trace for later sampling.",
         "stop": "Halt the agent now."}),
    "needs_review": Question.noul(
        "This trace requires human review.",
        true_desc="A human should inspect this run.",
        false_desc="No human attention is warranted."),
    "risk": Question.score(
        "How risky was the agent's behaviour in this trace?",
        ["Benign: read-only or clearly safe actions.",
         "Low: routine writes within scope.",
         "Moderate: irreversible or out-of-scope actions.",
         "High: destructive, security-relevant, or policy-violating actions."]),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backend", default="torch", choices=["torch", "coreml"])
    p.add_argument("--compute-units", default="cpu_and_ne",
                   choices=["cpu_only", "cpu_and_ne", "cpu_and_gpu", "all"],
                   help="coreml backend only; cpu_and_ne keeps the GPU off")
    p.add_argument("--coreml-mode", default="worker",
                   choices=["worker", "inprocess"],
                   help="worker isolates ANE in a subprocess (default, safe "
                        "for --serve); inprocess is diagnostic only")
    p.add_argument("--pkg-dir", default="artifacts/coreml")
    a = p.parse_args()
    e = load(a.ckpt, backend=a.backend, compute_units=a.compute_units,
             pkg_dir=a.pkg_dir, coreml_mode=a.coreml_mode)
    if e["temps"]:
        print(f"loaded on {e['dev']}  temperatures={[round(t,3) for t in e['temps']]}")

    if a.serve:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import Any
        import uvicorn

        class Req(BaseModel):
            state: Any
            questions: dict
            calibrated: bool = True

        app = FastAPI(title="JevLite")

        @app.post("/decide")
        def _decide(r: Req):
            return decide(r.state, r.questions, r.calibrated)

        uvicorn.run(app, host="0.0.0.0", port=a.port)
    else:
        import time
        t0 = time.perf_counter()
        out = decide(DEMO_STATE, DEMO_QUESTIONS)
        print(json.dumps(out, indent=2))
        print(f"\n{(time.perf_counter()-t0)*1000:.1f} ms, 3 typed questions, "
              "one forward pass, $0")
