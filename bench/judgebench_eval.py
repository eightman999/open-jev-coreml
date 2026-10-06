#!/usr/bin/env python3
"""JudgeBench external-gold evaluation for Jev / JevLite (/ control judges).

Each pair is judged twice with displayed order swapped (pass 1: A=orig A,
pass 2: A=orig B). Verdicts are mapped back to original response identity
before scoring against the JudgeBench gold label.

  python bench/judgebench_eval.py --system jev --out artifacts/judgebench_raw.jsonl
  python bench/judgebench_eval.py --system jevlite-coreml --out ...
  python bench/judgebench_eval.py --analyze-only --out artifacts/judgebench_raw.jsonl \
      --summary artifacts/judgebench_summary.json

`uncertain` is a valid explicit abstention only; transport/parse failures are
recorded as TIMEOUT/HTTP_ERROR/PARSE_FAILURE/REFUSAL and retried on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.judgebench_adapter import (  # noqa: E402
    JB_CRITERIA, JB_INSTR, JEV_MAX_STATE_CHARS, canonical_winner,
    gold_winner, jb_questions, jb_state)

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_RETRIES = 6
FAIL_TAG = ("TIMEOUT", "HTTP_ERROR", "PARSE_FAILURE", "REFUSAL")
RETRYABLE = {"TIMEOUT", "HTTP_ERROR"}


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- backends ----------------------------------------------------------------

def jev_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k
    return Path(os.path.expanduser("~/TYPESAFE_API_KEY")).read_text().strip()


def call_jev(state, questions, timeout):
    """-> (label, confidence, probabilities). Raises on transport failure."""
    body = {"model": JEV_MODEL,
            "state": state[:JEV_MAX_STATE_CHARS],
            "questions": questions}
    last = None
    for attempt in range(JEV_RETRIES):
        try:
            r = requests.post(JEV_URL, json=body,
                              headers={"Authorization":
                                       f"Bearer {jev_key()}"},
                              timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", 2.0))
                               * (2 ** attempt), 60.0))
                continue
            r.raise_for_status()
            ans = r.json()["answers"]["judge"]
            return (ans.get("choice", "uncertain"),
                    float(ans.get("confidence", 0.0)),
                    ans.get("probabilities"))
        except requests.RequestException as e:
            last = e
            if attempt + 1 < JEV_RETRIES:
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
    raise last


def call_llm(endpoint, model, prompt, api_key, temperature, timeout):
    """OpenAI-compatible control judge -> raw text."""
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": 512}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(f"{endpoint.rstrip('/')}/chat/completions",
                      json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("usage") or {}


REFUSAL_RE = __import__("re").compile(
    r"(i cannot|i can't|i can not|i'm unable|as an ai|cannot assist|"
    r"i must refuse|cannot evaluate|申し訳ありませんが|お答えできません)", __import__("re").I)


def llm_verdict(text):
    """Control-judge text -> label or failure tag."""
    import re as _re
    m = _re.search(r"\{.*\}", text, _re.S)
    if m:
        try:
            v = str(json.loads(m.group(0)).get("verdict", "")) \
                .strip().lower()
            if v in ("a", "candidate_a"):
                return "candidate_a"
            if v in ("b", "candidate_b"):
                return "candidate_b"
            if v in ("uncertain",):
                return "uncertain"
        except (json.JSONDecodeError, AttributeError):
            pass
    if REFUSAL_RE.search(text):
        return "REFUSAL"
    return "PARSE_FAILURE"


def llm_prompt(state):
    return (JB_INSTR + "\n\nReturn JSON only: "
            '{"verdict":"candidate_a|candidate_b|uncertain",'
            '"confidence":0.0,"reason":"..."}\n\n' + state)


# --- driver ------------------------------------------------------------------

def run_system(pairs, args, backend):
    done = set()
    if os.path.exists(args.out):
        for r in iter_jsonl(args.out):
            if r.get("verdict") in RETRYABLE:
                continue
            done.add((r.get("system"), r.get("pair_id"), r.get("pass")))

    work = [(p, ps) for p in pairs for ps in (1, 2)
            if (args.system, p["pair_id"], ps) not in done]
    print(f"{args.system}: {len(work)} pending calls", flush=True)

    lock = threading.Lock()
    out = open(args.out, "a", encoding="utf-8")
    n_done = [0]

    def one(item):
        pair, ps = item
        swapped = ps == 2
        state = jb_state(pair, swapped)
        rec = {"system": args.system, "pair_id": pair["pair_id"],
               "split": pair["split"], "source": pair.get("source"),
               "pass": ps, "swapped": swapped,
               "gold": gold_winner(pair["label"]),
               "state_chars": len(state),
               "state_chars_dropped":
                   max(0, len(state) - JEV_MAX_STATE_CHARS)
                   if args.system == "jev" else 0}
        t0 = time.perf_counter()
        try:
            if args.system == "jev":
                v, conf, probs = call_jev(state, jb_questions(),
                                          args.timeout)
            elif args.system.startswith("jevlite"):
                rec["ckpt"] = backend.get("ckpt")
                full_n, _ = backend["measure"](state, jb_questions())
                rec["input_tokens"] = full_n
                rec["truncated"] = full_n > backend["cap"]
                rec["tokens_dropped"] = max(0, full_n - backend["cap"])
                res = backend["engine"].decide(state, jb_questions())
                ans = res["judge"]
                v, conf, probs = (ans["label"], float(ans["confidence"]),
                                  ans.get("probabilities"))
            else:  # openai-compatible control judge
                raw, usage = call_llm(args.endpoint, args.model,
                                      llm_prompt(state), backend["key"],
                                      args.temperature, args.timeout)
                v = llm_verdict(raw)
                conf = 0.0
                probs = None
                rec["raw_response"] = raw
                rec["prompt_tokens"] = usage.get("prompt_tokens")
                rec["completion_tokens"] = usage.get("completion_tokens")
            rec.update(verdict=v, confidence=conf, probabilities=probs,
                       latency_s=round(time.perf_counter() - t0, 3))
        except requests.Timeout:
            rec.update(verdict="TIMEOUT", confidence=0.0)
        except requests.RequestException as e:
            rec.update(verdict="HTTP_ERROR", confidence=0.0,
                       reason=str(e)[:300])
        except Exception as e:  # noqa: BLE001 - engine failures are data
            rec.update(verdict="ERROR", confidence=0.0,
                       reason=f"{type(e).__name__}: {str(e)[:200]}")
        rec["canonical"] = (canonical_winner(rec["verdict"], swapped)
                            if rec["verdict"] not in FAIL_TAG else None)
        rec["correct"] = (rec["canonical"] == rec["gold"]
                          if rec["canonical"] in ("A", "B") else False)
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_done[0] += 1
            if n_done[0] % 50 == 0:
                print(f"  {args.system} {n_done[0]}/{len(work)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(one, work))
    else:
        for it in work:
            one(it)
    out.close()


# --- analysis ----------------------------------------------------------------

def _pct(a, b):
    return a / b if b else None


def _binary_forced(r):
    """argmax over candidate_a/candidate_b probabilities only (uncertain
    excluded) -> canonical winner. None when probabilities are absent."""
    p = r.get("probabilities")
    if not p:
        return None
    a, b = p.get("candidate_a", 0.0), p.get("candidate_b", 0.0)
    if a == b == 0.0:
        return None
    disp = "candidate_a" if a > b else "candidate_b"
    return canonical_winner(disp, r["swapped"])


def system_stats(rows, pairs):
    """Per-call and per-pair metrics for one system's rows.

    Denominators:
      overall_accuracy_abstain_wrong: all non-failure calls (uncertain = wrong)
      coverage: decided calls / non-failure calls
      accuracy_given_decision: correct decided / decided calls
      binary_forced_argmax_accuracy: calls where A/B argmax over stored
        probabilities is computable / calls with probabilities
    """
    valid = [r for r in rows if r["verdict"] not in FAIL_TAG]
    decisive = [r for r in valid if r["verdict"] in
                ("candidate_a", "candidate_b")]
    correct = [r for r in decisive if r["correct"]]
    fails = Counter(r["verdict"] for r in rows if r["verdict"] in FAIL_TAG)
    pos_a = sum(1 for r in decisive if r["verdict"] == "candidate_a")
    n1 = [r for r in valid if r["pass"] == 1]
    n2 = [r for r in valid if r["pass"] == 2]

    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["pass"]] = r
    strict = consistent = both_unc = flip_ab = flip_ba = 0
    dec2unc = unc2dec = 0
    for pid, pr in by_pair.items():
        if len(pr) < 2:
            continue
        c1, c2 = pr[1].get("canonical"), pr[2].get("canonical")
        strict += bool(c1 and c1 == pr[1]["gold"]
                       and c2 and c2 == pr[2]["gold"])
        if c1 == c2 and c1 is not None:
            consistent += 1
        if c1 == "uncertain" and c2 == "uncertain":
            both_unc += 1
        if c1 in ("A", "B") and c2 == "uncertain":
            dec2unc += 1
        if c1 == "uncertain" and c2 in ("A", "B"):
            unc2dec += 1
        if c1 == "A" and c2 == "B":
            flip_ab += 1
        if c1 == "B" and c2 == "A":
            flip_ba += 1

    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    confb = Counter()
    for r in valid:
        b = min(int(r["confidence"] * 5), 4) * 0.2
        key = f"{b:.1f}-{b + 0.2:.1f}"
        confb.setdefault(key, {"n": 0, "correct": 0})
        confb[key]["n"] += 1
        confb[key]["correct"] += bool(r["correct"])
    conf_buckets = {k: {"n": v["n"],
                        "overall_accuracy_abstain_wrong":
                            _pct(v["correct"], v["n"])}
                    for k, v in sorted(confb.items())}

    tb = {"<=512": [], "513-1024": [], ">1024": [], "truncated": []}
    for r in rows:
        if r["verdict"] in FAIL_TAG:
            continue
        n = r.get("input_tokens")
        if n is None:
            continue
        tb["<=512" if n <= 512 else "513-1024" if n <= 1024 else ">1024"] \
            .append(r)
        if r.get("truncated"):
            tb["truncated"].append(r)

    def acc(sub):
        d = [r for r in sub if r["verdict"] in ("candidate_a", "candidate_b")]
        return _pct(sum(r["correct"] for r in d), len(d)) \
            if d else None

    def oacc(sub):
        return _pct(sum(r["correct"] for r in sub), len(sub)) \
            if sub else None

    trunc = {k: {"n": len(v),
                 "overall_accuracy_abstain_wrong": oacc(v) if v else None,
                 "accuracy_given_decision": acc(v) if v else None,
                 "coverage": _pct(
                     sum(1 for r in v if r["verdict"]
                         in ("candidate_a", "candidate_b")), len(v))}
             for k, v in tb.items()}

    forced = [_binary_forced(r) for r in valid]
    forced_n = [w for w in forced if w is not None]
    forced_ok = sum(1 for r, w in zip(valid, forced)
                    if w is not None and w == r["gold"])

    return {
        "n_calls": len(rows), "n_valid": len(valid),
        "overall_accuracy_abstain_wrong": _pct(len(correct), len(valid)),
        "coverage": _pct(len(decisive), len(valid)),
        "accuracy_given_decision": _pct(len(correct), len(decisive)),
        "binary_forced_argmax_accuracy": _pct(forced_ok, len(forced_n)),
        "binary_forced_argmax_n": len(forced_n),
        "selective_risk": (1 - _pct(len(correct), len(decisive))
                           if decisive else None),
        "abstention_rate": _pct(
            sum(1 for r in valid if r["verdict"] == "uncertain"),
            len(valid)),
        "wrong_answer_rate": _pct(
            sum(1 for r in decisive if not r["correct"]), len(valid)),
        "first_position_rate_decided": _pct(pos_a, len(decisive)),
        "strict_pair_accuracy": _pct(strict, len(by_pair)),
        "swap_consistency": _pct(consistent, len(by_pair)),
        "consistency_detail": {
            "both_uncertain": both_unc, "decisive_to_uncertain": dec2unc,
            "uncertain_to_decisive": unc2dec,
            "flip_A_to_B": flip_ab, "flip_B_to_A": flip_ba},
        "failures": dict(fails),
        "latency": {"mean": float(np.mean(lat)) if lat else None,
                    "p50": float(np.percentile(lat, 50)) if lat else None,
                    "p95": float(np.percentile(lat, 95)) if lat else None},
        "confidence_buckets": conf_buckets,
        "truncation": trunc,
        "original_order_accuracy": _pct(
            sum(r["correct"] for r in n1), len(n1)),
        "swapped_order_accuracy": _pct(
            sum(r["correct"] for r in n2), len(n2)),
    }


def analyze(rows, pairs, args):
    systems = sorted({r["system"] for r in rows})
    per_system = {}
    for sname in systems:
        sub = [r for r in rows if r["system"] == sname]
        st = system_stats(sub, pairs)
        for split in ("gpt", "claude"):
            ssub = [r for r in sub if r["split"] == split]
            st.setdefault("per_split", {})[split] = {
                "n_calls": len(ssub),
                "overall_accuracy_abstain_wrong": _pct(
                    sum(r["correct"] for r in ssub
                        if r["verdict"] in ("candidate_a", "candidate_b")),
                    len([r for r in ssub
                         if r["verdict"] not in FAIL_TAG])),
                "coverage": _pct(
                    sum(1 for r in ssub if r["verdict"]
                        in ("candidate_a", "candidate_b")),
                    len([r for r in ssub
                         if r["verdict"] not in FAIL_TAG]))}
        src = {}
        for r in sub:
            src.setdefault(r.get("source") or "unknown", []).append(r)
        st["per_source"] = {
            k: {"n_calls": len(v),
                "overall_accuracy_abstain_wrong": _pct(
                    sum(r["correct"] for r in v
                        if r["verdict"] in ("candidate_a", "candidate_b")),
                    len([r for r in v if r["verdict"] not in FAIL_TAG])),
                "coverage": _pct(
                    sum(1 for r in v if r["verdict"]
                        in ("candidate_a", "candidate_b")),
                    len([r for r in v
                         if r["verdict"] not in FAIL_TAG])),
                "low_n": len(v) < 20}
            for k, v in sorted(src.items())}
        per_system[sname] = st

    matrices = {}
    disagreements = []
    if "jev" in per_system and any(s.startswith("jevlite")
                                 for s in per_system):
        jl = next(s for s in systems if s.startswith("jevlite"))
        jr = {}
        lr = {}
        for r in rows:
            if r["system"] == "jev":
                jr.setdefault(r["pair_id"], {})[r["pass"]] = r
            elif r["system"] == jl:
                lr.setdefault(r["pair_id"], {})[r["pass"]] = r

        def _single(jr_, lr_):
            j_ok = (jr_["canonical"] in ("A", "B")
                    and jr_["canonical"] == jr_["gold"])
            l_ok = (lr_["canonical"] in ("A", "B")
                    and lr_["canonical"] == lr_["gold"])
            l_abs = lr_["canonical"] == "uncertain"
            if j_ok and l_ok:
                return "both_correct"
            if j_ok and not l_ok and not l_abs:
                return "jev_only"
            if l_ok and not j_ok:
                return "jevlite_only"
            if l_abs and j_ok:
                return "jl_abstain_jev_correct"
            if l_abs and not j_ok:
                return "jl_abstain_jev_wrong"
            return "both_wrong"

        def _pair_state(pr_):
            """Collapse a system's two passes into a pair-level state:
            correct | wrong | abstain | mixed."""
            cans = [pr_[ps]["canonical"] for ps in (1, 2) if ps in pr_]
            if not cans:
                return "mixed"
            if all(c == "uncertain" for c in cans):
                return "abstain"
            if any(c in (None,) or c not in ("A", "B", "uncertain")
                   for c in cans):
                return "mixed"
            if all(c in ("A", "B") for c in cans):
                oks = [pr_[ps]["canonical"] == pr_[ps]["gold"]
                       for ps in (1, 2) if ps in pr_]
                if all(oks):
                    return "correct"
                return "wrong" if not any(oks) else "mixed"
            return "mixed"

        mo, ms, mt, mt_detail = Counter(), Counter(), Counter(), Counter()
        for p in pairs:
            pid = p["pair_id"]
            a1, b1 = jr.get(pid, {}).get(1), lr.get(pid, {}).get(1)
            a2, b2 = jr.get(pid, {}).get(2), lr.get(pid, {}).get(2)
            if a1 and b1:
                mo[_single(a1, b1)] += 1
            if a2 and b2:
                ms[_single(a2, b2)] += 1
            if pid in jr and pid in lr:
                js, ls = _pair_state(jr[pid]), _pair_state(lr[pid])
                mt_detail[f"jev={js}+jl={ls}"] += 1
                if js == "correct" and ls == "correct":
                    mt["both_correct"] += 1
                elif js == "correct" and ls == "wrong":
                    mt["jev_only"] += 1
                elif ls == "correct" and js != "correct":
                    mt["jevlite_only"] += 1
                elif ls == "abstain" and js == "correct":
                    mt["jl_abstain_jev_correct"] += 1
                elif ls == "abstain" and js == "wrong":
                    mt["jl_abstain_jev_wrong"] += 1
                elif js == "wrong" and ls == "wrong":
                    mt["both_wrong"] += 1
                else:
                    mt["mixed_or_inconsistent"] += 1
            if a1 and b1 and (a1["canonical"] != b1["canonical"]
                              or b1["canonical"] == "uncertain"):
                disagreements.append({
                    "pair_id": pid, "source": p.get("source"),
                    "split": p["split"],
                    "question": p.get("question"), "gold": p["label"],
                    "jev": {f"pass{ps}": {"verdict": jr[pid][ps]["verdict"],
                                          "canonical": jr[pid][ps]
                                          ["canonical"],
                                          "confidence": jr[pid][ps]
                                          ["confidence"]}
                            for ps in (1, 2) if ps in jr[pid]},
                    "jevlite": {f"pass{ps}": {"verdict": lr[pid][ps]
                                              ["verdict"],
                                              "canonical": lr[pid][ps]
                                              ["canonical"],
                                              "confidence": lr[pid][ps]
                                              ["confidence"]}
                                for ps in (1, 2) if ps in lr[pid]}})
        matrices = {"outcome_matrix_original": dict(mo),
                    "outcome_matrix_swapped": dict(ms),
                    "outcome_matrix_strict_pair": dict(mt),
                    "outcome_matrix_strict_pair_detail": dict(mt_detail)}

    return {"n_pairs": len(pairs), "systems": per_system,
            **matrices}, disagreements


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse
                                .RawDescriptionHelpFormatter)
    p.add_argument("--data", default="artifacts/judgebench_620.jsonl")
    p.add_argument("--system", required=True,
                   help="jev | jevlite-coreml | jevlite-torch | "
                        "<openai-compatible name>")
    p.add_argument("--out", default="artifacts/judgebench_raw.jsonl")
    p.add_argument("--summary", default="artifacts/judgebench_summary.json")
    p.add_argument("--disagreements",
                   default="artifacts/judgebench_disagreements.jsonl")
    p.add_argument("--failures",
                   default="artifacts/judgebench_failures.jsonl")
    p.add_argument("--endpoint", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--api-key-env", default="LLM_JUDGE_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ckpt", default="jevlite.pt",
                   help="checkpoint for jevlite-* systems")
    p.add_argument("--slots", default=None,
                   choices=["none", "lead", "pair"],
                   help="override the checkpoint's label serialization; "
                        "default: whatever the checkpoint was trained with "
                        "(ck['slots'], falling back to 'lead')")
    p.add_argument("--subset-seed", type=int, default=None,
                   help="when set, the --limit subset is a seeded random "
                        "sample of pairs instead of the first N")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    pairs = list(iter_jsonl(args.data))
    subset_method = "head"
    if args.subset_seed is not None:
        import random
        pairs = random.Random(args.subset_seed).sample(
            pairs, args.limit or len(pairs))
        subset_method = f"seeded-sample(seed={args.subset_seed})"
    elif args.limit:
        pairs = pairs[:args.limit]

    backend = {}
    if not args.analyze_only:
        if args.system == "jev":
            jev_key()  # fail fast if missing
        elif args.system.startswith("jevlite"):
            backend_name = ("coreml" if "coreml" in args.system
                            else "torch")
            kwargs = {}
            if backend_name == "torch":
                kwargs["device"] = "mps"
            from engine import DecisionEngine
            backend["engine"] = DecisionEngine(backend=backend_name,
                                               ckpt=args.ckpt,
                                               slots=args.slots, **kwargs)
            backend["ckpt"] = args.ckpt
            # token + truncation stats on the same encoding path
            import torch
            from benchmark_jev_competition import make_stats_encoder
            ck = torch.load(args.ckpt, map_location="cpu",
                            weights_only=False)
            backend["slots"] = args.slots or ck.get("slots", "lead")
            backend["cap"] = getattr(backend["engine"]._impl,
                                     "encode_len", ck.get("max_len", 1024))
            backend["measure"] = make_stats_encoder(
                ck["encoder"], 1 << 22, slots=backend["slots"])
        else:
            if not args.endpoint or not args.model:
                p.error("control judge needs --endpoint and --model")
            backend["key"] = os.environ.get(args.api_key_env)
        run_system(pairs, args, backend)
        if backend.get("engine"):
            backend["engine"].close()

    rows = list(iter_jsonl(args.out)) if os.path.exists(args.out) else []
    summary, disagreements = analyze(rows, pairs, args)
    pair_ids = [p["pair_id"] for p in pairs]
    summary["meta"] = {
        "dataset": "ScalerLab/JudgeBench",
        "dataset_sha256": sha256_file(args.data),
        "pairs": len(pairs),
        "subset": {
            "method": subset_method,
            "seed": args.subset_seed,
            "n": len(pairs),
            "pair_ids": pair_ids,
            "sha256": hashlib.sha256(
                "\n".join(pair_ids).encode()).hexdigest()},
        "ckpt": args.ckpt if args.system.startswith("jevlite") else None,
        "slots": backend.get("slots") or args.slots,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "host": platform.node(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "JudgeBench label is the only gold; Jev is a system "
                "under test, not a reference."}
    json.dump(summary, open(args.summary, "w"), indent=1,
              ensure_ascii=False)

    if args.disagreements:
        with open(args.disagreements, "w", encoding="utf-8") as f:
            for d in disagreements:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if args.failures:
        with open(args.failures, "w", encoding="utf-8") as f:
            for r in rows:
                if r.get("verdict") in FAIL_TAG:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for sname, st in summary["systems"].items():
        print(f"\n== {sname}: acc={st['overall_accuracy_abstain_wrong']} "
              f"cov={st['coverage']} cond={st['accuracy_given_decision']} "
              f"forced_ab={st['binary_forced_argmax_accuracy']} "
              f"strict={st['strict_pair_accuracy']} "
              f"swap={st['swap_consistency']} fails={st['failures']}")
    for k in ("outcome_matrix_original", "outcome_matrix_swapped",
              "outcome_matrix_strict_pair"):
        if k in summary:
            print(k + ":", summary[k])
    print("wrote", args.summary)


if __name__ == "__main__":
    raise SystemExit(main())
