"""JevLite never-A root-cause audit.

Traces why `candidate_a` never wins on JudgeBench (0/1240 calls) by checking
each stage independently of the CoreML/worker path:

  D1  training label distribution   -> artifacts/jevlite_label_audit.json
  D2  label <-> position mapping    -> same file, "position_mapping" section
  D3  shared score head + per-label logit stats -> jevlite_head_audit.json
  D4  synthetic sanity suite        -> jevlite_synthetic_audit.jsonl
  D5  class-order permutation test  -> jevlite_permutation_audit.jsonl
  D7  short-input replication       -> covered by D4/D5 (all inputs << 512 tok)

JevLite scores every <<l>> marker with ONE shared Linear(d,1) head, so a
fixed per-class head bias cannot exist; the audit reframes "class index" as
position-within-sorted-criteria.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.judgebench_adapter import (  # noqa: E402
    JB_CRITERIA, JB_INSTR, jb_state)

ART = "artifacts"


# --- D1: training label distribution ------------------------------------------

def label_audit(limit=None):
    """LocalLLaMA/typed-decisions label stats per sorted position."""
    import td_data
    from typed_schema import iter_labels

    out = {"dataset": td_data.DATASET, "splits": {}}
    for split in ("train", "validation", "test"):
        try:
            rows = td_data.load_split(split, "all", limit=limit)
        except Exception as e:  # noqa: BLE001 - missing splits are data
            out["splits"][split] = {"error": f"{type(e).__name__}: {e}"}
            continue
        per_type = defaultdict(lambda: {
            "n_questions": 0, "n_labels": 0,
            "label_names": Counter(),
            "pos_targets": defaultdict(list),
            "pos_argmax": Counter(),
        })
        for r in rows:
            for qname, q in r["questions"].items():
                labs = iter_labels(q)
                probs = (r.get("gold", {}).get(qname, {})
                         .get("probabilities", {}))
                tgt = [float(probs.get(l, 0.0)) for l, _ in labs]
                t = per_type[q["type"]]
                t["n_questions"] += 1
                t["n_labels"] += len(labs)
                for i, (l, _) in enumerate(labs):
                    t["label_names"][l] += 1
                    t["pos_targets"][i].append(tgt[i])
                if tgt and max(tgt) > 0:
                    for i, v in enumerate(tgt):
                        if v == max(tgt):
                            t["pos_argmax"][i] += 1
                            break
        s = {}
        for qt, t in per_type.items():
            s[qt] = {
                "n_questions": t["n_questions"],
                "n_labels": t["n_labels"],
                "top_label_names": t["label_names"].most_common(15),
                "mean_target_by_position": {
                    str(k): sum(v) / len(v)
                    for k, v in sorted(t["pos_targets"].items())},
                "argmax_count_by_position": dict(
                    sorted(t["pos_argmax"].items())),
                "has_uncertain_label": bool(
                    t["label_names"].get("uncertain")),
            }
        out["splits"][split] = {"n_rows": len(rows), "per_type": s}
    return out


# --- D2: label <-> position mapping ---------------------------------------------

def position_mapping(questions, ck):
    """The pipeline has no class index: sorted criteria order IS the index."""
    import td_data
    from typed_schema import iter_labels
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ck["encoder"])
    _, qid, lid = td_data.add_markers(tok)
    row = {"id": "map", "state": "s", "questions": questions, "gold": {}}
    enc = td_data.encode(row, tok, ck.get("max_len", 1024), qid, lid,
                         with_gold=False)
    stages, off = [], 0
    for gi, qname in enumerate(enc["qnames"]):
        for i, (lab, desc) in enumerate(iter_labels(questions[qname])):
            stages.append({
                "question": qname, "sorted_pos": i, "label": lab,
                "label_text": f"{lab}: {desc}"[:60],
                "encode_label_pos": enc["label_pos"][off + i],
                "group": gi,
                "labels_list_entry": enc["labels"][gi][i],
                "decide_key": lab})
        off += len(enc["labels"][gi])
    return {"note": "no class index exists; sorted criteria position is the "
                    "index end-to-end (iter_labels -> encode label_pos -> "
                    "gather -> softmax -> decide dict key)",
            "stages": stages}


# --- torch inference with raw logits ---------------------------------------------

def torch_logits(state, questions, eng):
    """-> (logits[M], probs[M], labels per group) on the PyTorch model."""
    import torch
    import td_data
    m = eng["model"]
    row = {"id": "live", "state": state, "questions": questions, "gold": {}}
    enc = td_data.encode(row, m.tok, eng["max_len"], m.qid, m.lid,
                         with_gold=False,
                         slots=eng.get("slots", "lead"))
    b = td_data.collate([enc], m.tok.pad_token_id)
    b = {k: v.to(eng["dev"]) for k, v in b.items()}
    with torch.no_grad():
        lg, probs = m(b, apply_temperature=True)
    return (lg[0].cpu().tolist(), probs[0].cpu().tolist(), enc)


def verdict_of(probs, enc, gi=0):
    labels = enc["labels"][gi]
    d = dict(zip(labels, probs[:len(labels)]))
    top = max(d, key=d.get)
    return {"label": top, "confidence": d[top],
            "probabilities": {k: round(v, 4) for k, v in d.items()}}


# --- D3: score head + per-label logit stats ----------------------------------------

def head_audit(eng, pairs, n=60):
    import statistics as st
    m = eng["model"]
    head = m.score
    w = head.weight.detach().cpu()
    info = {"shared_head": True,
            "note": "single Linear(d,1) scores every <<l>> position; "
                    "per-class head collapse is structurally impossible",
            "weight_shape": list(w.shape),
            "bias": float(head.bias.detach().cpu()[0]),
            "weight_l2": float(w.norm()),
            "weight_mean": float(w.mean()),
            "weight_std": float(w.std())}
    per_label = defaultdict(list)
    for p in pairs[:n]:
        state = jb_state(p, swapped=False)
        lg, _, enc = torch_logits(state, jb_questions(), eng)
        for i, lab in enumerate(enc["labels"][0]):
            per_label[lab].append(lg[i])
    stats = {lab: {"n": len(v), "mean": st.mean(v),
                   "median": st.median(v), "max": max(v),
                   "min": min(v)}
             for lab, v in sorted(per_label.items())}
    return {"head": info, "logit_stats_on_judgebench": stats}


def jb_questions():
    return {"judge": {"type": "choice", "instructions": JB_INSTR,
                      "criteria": dict(JB_CRITERIA)}}


# --- D4+D7: synthetic sanity suite ---------------------------------------------------

A_TXT = "candidate A is objectively more correct"
B_TXT = "candidate B is objectively more correct"
U_TXT = "cannot decide reliably from the supplied information"

SYNTH = [
    # obvious-A cases
    {"id": "obvA1", "q": "What is 2+2?",
     "a": "2+2 equals 4.", "b": "2+2 equals 5.", "want": "candidate_a"},
    {"id": "obvA2", "q": "What is the capital of France?",
     "a": "The capital of France is Paris.", "b": "The capital of France is Madrid.", "want": "candidate_a"},
    {"id": "obvA3", "q": "Is water made of hydrogen and oxygen?",
     "a": "Yes, water is H2O.", "b": "No, water is CO2.", "want": "candidate_a"},
    {"id": "obvA4", "q": "Which is larger: 10 or 3?",
     "a": "10 is larger than 3.", "b": "3 is larger than 10.", "want": "candidate_a"},
    {"id": "obvA5", "q": "What color is the sky on a clear day?",
     "a": "The sky is blue.", "b": "The sky is orange.", "want": "candidate_a"},
    {"id": "obvA6", "q": "How many days are in a week?",
     "a": "There are 7 days in a week.", "b": "There are 9 days in a week.", "want": "candidate_a"},
    # obvious-B cases (mirror of A: the correct content sits in B)
    {"id": "obvB1", "q": "What is 2+2?",
     "a": "2+2 equals 5.", "b": "2+2 equals 4.", "want": "candidate_b"},
    {"id": "obvB2", "q": "What is the capital of France?",
     "a": "The capital of France is Madrid.", "b": "The capital of France is Paris.", "want": "candidate_b"},
    {"id": "obvB3", "q": "Is water made of hydrogen and oxygen?",
     "a": "No, water is CO2.", "b": "Yes, water is H2O.", "want": "candidate_b"},
    {"id": "obvB4", "q": "Which is larger: 10 or 3?",
     "a": "3 is larger than 10.", "b": "10 is larger than 3.", "want": "candidate_b"},
    {"id": "obvB5", "q": "What color is the sky on a clear day?",
     "a": "The sky is orange.", "b": "The sky is blue.", "want": "candidate_b"},
    {"id": "obvB6", "q": "How many days are in a week?",
     "a": "There are 9 days in a week.", "b": "There are 7 days in a week.", "want": "candidate_b"},
    # identical candidates
    {"id": "same1", "q": "What is 2+2?", "a": "2+2 equals 4.",
     "b": "2+2 equals 4.", "want": "uncertain"},
    {"id": "same2", "q": "What is the capital of France?",
     "a": "Paris is the capital.", "b": "Paris is the capital.", "want": "uncertain"},
    # both wrong
    {"id": "bw1", "q": "What is 2+2?", "a": "2+2 equals 5.",
     "b": "2+2 equals 7.", "want": "uncertain"},
    {"id": "bw2", "q": "What is the capital of France?",
     "a": "The capital is Madrid.", "b": "The capital is Rome.", "want": "uncertain"},
    # length-only difference, same content
    {"id": "len1", "q": "What is 2+2?", "a": "2+2 equals 4. " * 40 + "That is the answer.",
     "b": "2+2 equals 4.", "want": None},
    {"id": "len2", "q": "What is 2+2?", "a": "2+2 equals 4.",
     "b": "2+2 equals 4. " * 40 + "That is the answer.", "want": None},
    # wording-only difference
    {"id": "word1", "q": "What is 2+2?", "a": "2+2 equals 4.",
     "b": "The sum of 2 and 2 is 4.", "want": None},
    {"id": "word2", "q": "What is the capital of France?",
     "a": "Paris.", "b": "The capital of France is Paris.", "want": None},
]


def synth_state(c, swapped):
    a, b = (c["b"], c["a"]) if swapped else (c["a"], c["b"])
    return f"QUESTION:\n{c['q']}\n\nCANDIDATE A:\n{a}\n\nCANDIDATE B:\n{b}"


def synthetic_suite(teng, ceng):
    rows = []
    for c in SYNTH:
        for ps in (1, 2):
            sw = ps == 2
            st = synth_state(c, sw)
            rec = {"id": c["id"], "pass": ps, "swapped": sw,
                   "expected_display": (
                       {"candidate_a": "candidate_b",
                        "candidate_b": "candidate_a"}.get(c["want"], c["want"])
                       if sw else c["want"])}
            lg, probs, enc = torch_logits(st, jb_questions(), teng)
            v = verdict_of(probs, enc)
            rec["torch"] = {"logits": [round(x, 4) for x in lg],
                            "verdict": v["label"],
                            "probabilities": v["probabilities"]}
            if ceng is not None:
                cv = ceng.decide(st, jb_questions())
                rec["coreml"] = {"verdict": cv["judge"]["label"],
                                 "probabilities": cv["judge"]
                                 ["probabilities"]}
            rows.append(rec)
    return rows


# --- D5: permutation test --------------------------------------------------------

PERM_LAYOUTS = {
    "A-B-U": {"aa": A_TXT, "bb": B_TXT, "cc": U_TXT},
    "B-A-U": {"aa": B_TXT, "bb": A_TXT, "cc": U_TXT},
    "U-A-B": {"aa": U_TXT, "bb": A_TXT, "cc": B_TXT},
    "A-U-B": {"aa": A_TXT, "bb": U_TXT, "cc": B_TXT},
}


def perm_questions(criteria):
    return {"judge": {"type": "choice", "instructions": JB_INSTR,
                      "criteria": dict(criteria)}}


def permutation_test(teng, ceng, cases=None):
    """Same meanings at different sorted positions: does suppression follow
    the position index or the 'candidate A wins' meaning?"""
    cases = cases or [c for c in SYNTH if c["want"] == "candidate_a"]
    rows = []
    for lname, crit in PERM_LAYOUTS.items():
        for c in cases:
            for ps in (1, 2):
                sw = ps == 2
                st = synth_state(c, sw)
                rec = {"layout": lname, "id": c["id"], "pass": ps,
                       "swapped": sw,
                       "pos_meaning": {str(i): lab for i, lab in
                                       enumerate(sorted(crit))}}
                lg, probs, enc = torch_logits(st, perm_questions(crit), teng)
                v = verdict_of(probs, enc)
                rec["torch"] = {"logits": [round(x, 4) for x in lg],
                                "verdict": v["label"],
                                "probabilities": v["probabilities"],
                                "verdict_pos": enc["labels"][0]
                                .index(v["label"])}
                if ceng is not None:
                    cv = ceng.decide(st, perm_questions(crit))
                    rec["coreml"] = {
                        "verdict": cv["judge"]["label"],
                        "probabilities": cv["judge"]["probabilities"]}
                rows.append(rec)
    return rows


def summarize_perm(rows):
    """Per layout: mean prob and win rate per sorted position."""
    agg = {}
    for lname in PERM_LAYOUTS:
        sub = [r for r in rows if r["layout"] == lname]
        pos_probs = defaultdict(list)
        wins = Counter()
        for r in sub:
            pr = r["torch"]["probabilities"]
            for i, lab in enumerate(sorted(pr)):
                pos_probs[i].append(pr[lab])
            wins[r["torch"]["verdict_pos"]] += 1
        agg[lname] = {
            "pos_mean_prob": {str(k): sum(v) / len(v)
                              for k, v in sorted(pos_probs.items())},
            "win_count_by_pos": dict(sorted(wins.items())),
            "n": len(sub)}
    return agg


# --- main -------------------------------------------------------------------------

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=f"{ART}/judgebench_620.jsonl")
    ap.add_argument("--head-n", type=int, default=60)
    ap.add_argument("--label-limit", type=int, default=0)
    ap.add_argument("--no-coreml", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma list: labels,mapping,head,synth,perm")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else {
        "labels", "mapping", "head", "synth", "perm"}

    os.makedirs(ART, exist_ok=True)
    pairs = list(iter_jsonl(a.pairs)) if os.path.exists(a.pairs) else []

    teng = ceng = None
    if only & {"head", "synth", "perm"}:
        import importlib
        serve = importlib.import_module("05_serve")
        teng = serve.load("jevlite.pt")
        if not a.no_coreml:
            from coreml_worker import WorkerCoreMLEngine
            ceng = WorkerCoreMLEngine("artifacts/coreml", "jevlite.pt",
                                      "cpu_and_ne")

    if only & {"labels", "mapping"}:
        la = label_audit(limit=a.label_limit or None)
        import torch
        ck = torch.load("jevlite.pt", map_location="cpu",
                        weights_only=False)
        la["position_mapping"] = position_mapping(jb_questions(), ck)
        json.dump(la, open(f"{ART}/jevlite_label_audit.json", "w"),
                  indent=1, ensure_ascii=False)
        print("wrote jevlite_label_audit.json")
        for split, s in la["splits"].items():
            print(" ", split, json.dumps(s.get("per_type"), default=str)[:300])

    if "head" in only:
        ha = head_audit(teng, pairs, n=a.head_n)
        json.dump(ha, open(f"{ART}/jevlite_head_audit.json", "w"),
                  indent=1, ensure_ascii=False)
        print("wrote jevlite_head_audit.json")
        print(" ", json.dumps(ha["logit_stats_on_judgebench"], indent=1))

    if "synth" in only:
        rows = synthetic_suite(teng, ceng)
        with open(f"{ART}/jevlite_synthetic_audit.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tv = Counter(r["torch"]["verdict"] for r in rows)
        cv = Counter(r["coreml"]["verdict"] for r in rows if "coreml" in r)
        print(f"synthetic: torch {dict(tv)} | coreml {dict(cv)}")

    if "perm" in only:
        rows = permutation_test(teng, ceng)
        with open(f"{ART}/jevlite_permutation_audit.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("permutation summary:",
              json.dumps(summarize_perm(rows), indent=1))

    if ceng is not None:
        ceng.close()


if __name__ == "__main__":
    main()
