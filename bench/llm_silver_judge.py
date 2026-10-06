"""LLM silver-judge: anonymous A/B evaluation of historical-Jev vs JevLite.

Evaluates which of the two systems' verdicts on the 400-case stratified
sample (artifacts/jev_human_review.jsonl) is actually justified by the
task definition - with neither system's identity revealed to the judge.

Blinding rules (do not break these):
  - the judge prompt never names Jev / JevLite / CoreML / ANE;
  - candidates are candidate_a / candidate_b only;
  - per-case base order is deterministic via sha256 (never Python hash());
  - pass 2 fully inverts the order so position bias is measurable;
  - refusals are counted separately and never folded into AMBIGUOUS.

Silver labels are LLM-generated - not human gold.

  python bench/llm_silver_judge.py \
      --review-file artifacts/jev_human_review.jsonl \
      --input /path/to/queue_*.resplit.jsonl \
      --endpoint http://HOST:PORT/v1 --model NAME \
      --output artifacts/jev_llm_judge_raw.jsonl --passes 2

Append-only output -> interrupt and rerun freely; done (queue_id, pass)
pairs are skipped. --analyze-only rebuilds all aggregates from an existing
raw file without calling the API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_jev_competition import (  # noqa: E402
    CRITERIA, INSTR, VERDICTS, iter_jsonl, load_inputs, jev_state,
    jevlite_questions, make_stats_encoder, sha256_file)

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "prompts", "llm_silver_judge_v1.txt")
SWAP_SALT = "silver-judge-v1"
VERDICTS_5 = ("A", "B", "BOTH", "NEITHER", "AMBIGUOUS")
CONF_THRESHOLD = 0.75

REFUSAL_RE = re.compile(
    r"(i cannot|i can't|i can not|i'm unable|as an ai|cannot assist|"
    r"i must refuse|cannot evaluate|inappropriate|申し訳ありませんが|"
    r"お答えできません)", re.I)


def case_swap(queue_id: str) -> bool:
    """Deterministic per-case base order. Never Python hash()."""
    return int(hashlib.sha256(
        f"{queue_id}:{SWAP_SALT}".encode()).hexdigest(), 16) % 2 == 0


def render_prompt(template: str, task, input_text, state, choices,
                  cand_a, cand_b) -> str:
    """Token replacement, not str.format - state text may contain braces."""
    out = template
    for tok, val in (("{task}", task), ("{input_text}", input_text),
                     ("{state}", state), ("{choices}", choices),
                     ("{candidate_a}", cand_a), ("{candidate_b}", cand_b)):
        out = out.replace(tok, val)
    return out


def case_fields(cand):
    """What the judge sees as INPUT - the workload's own header fields plus
    the actual marked-span text (the same bytes the span offsets point to)."""
    span = cand.get("span") or {}
    lines = [
        f"対象レコード: {cand.get('record_id', '')} "
        f"(category={cand.get('category', '')})",
        f"確認事項: {cand.get('question', '')}",
        f"候補区間: 文字 {span.get('start', '?')}〜{span.get('end', '?')}",
        f"候補区間のテキスト: {cand.get('text', '')}",
    ]
    return "\n".join(lines)


def choices_text():
    return "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())


def order_for(queue_id, pass_no):
    """pass 1: base order (half the cases swapped). pass 2: full inversion."""
    order = ["jev", "jevlite"] if not case_swap(queue_id) \
        else ["jevlite", "jev"]
    if pass_no == 2:
        order = order[::-1]
    return order


def call_judge(endpoint, model, prompt, api_key, temperature, timeout,
               json_mode):
    """One chat/completions call -> (raw_text, usage, latency_s)."""
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": 1024}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    r = requests.post(f"{endpoint.rstrip('/')}/chat/completions",
                      json=body, headers=headers, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    return msg, data.get("usage") or {}, dt


def parse_verdict(text):
    """-> (verdict, confidence, reason) or raises ValueError.

    Strict JSON first; on failure, salvage the three fields by regex.
    Some judges emit unescaped quotes inside string values (e.g. a reason
    containing "noise"), which breaks strict parsing despite being a clear
    verdict."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    frag = m.group(0)
    try:
        obj = json.loads(frag)
        v = str(obj.get("verdict", "")).strip().upper()
        if v not in VERDICTS_5:
            raise ValueError(f"bad verdict {v!r}")
        return v, float(obj.get("confidence", 0.0)), str(
            obj.get("reason", ""))
    except json.JSONDecodeError:
        mv = re.search(r'"verdict"\s*:\s*"([A-Za-z]+)"', frag)
        mc = re.search(r'"confidence"\s*:\s*([0-9.]+)', frag)
        mr = re.search(r'"reason"\s*:\s*"(.*)"\s*\}?\s*$', frag, re.S)
        if not mv:
            raise ValueError("no verdict field")
        v = mv.group(1).strip().upper()
        if v not in VERDICTS_5:
            raise ValueError(f"bad verdict {v!r}")
        return v, float(mc.group(1)) if mc else 0.0, \
            mr.group(1) if mr else ""


def classify(text):
    """-> verdict string or failure tag.

    Parse first: a well-formed verdict JSON is never a refusal, even if the
    quoted candidate text itself contains apology-like phrases (judge cases
    routinely quote 「申し訳ありませんが...」 author notes)."""
    try:
        return parse_verdict(text)
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        if REFUSAL_RE.search(text):
            return "REFUSAL", 0.0, ""
        return "PARSE_FAILURE", 0.0, str(e)


RETRYABLE = {"TIMEOUT", "HTTP_ERROR"}


def load_done(path):
    """Completed (queue_id, pass) pairs. Transport-level failures are
    retried on resume; only terminal verdicts count as done."""
    done = set()
    if os.path.exists(path):
        for r in iter_jsonl(path):
            if r.get("verdict") in RETRYABLE:
                continue
            done.add((r.get("queue_id"), r.get("pass")))
    return done


def run_judge(cases, cand_by_id, template, args):
    """Sequential or threaded (--concurrency) calls; append-only output.

    Row order in the raw file is nondeterministic under concurrency - all
    downstream analysis keys on (queue_id, pass), so results are identical.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = load_done(args.output)
    work = []
    for case in cases:
        qid = case["queue_id"]
        cand = cand_by_id.get(qid, {})
        state = jev_state(cand) if cand else case.get("context", "")
        input_text = case_fields(cand) if cand else \
            f"record: {case.get('record_id', '')}"
        verdicts = {"jev": case["jev_verdict"],
                    "jevlite": case["jevlite_verdict"]}
        for p in range(1, args.passes + 1):
            if (qid, p) in done:
                continue
            order = order_for(qid, p)
            prompt = render_prompt(
                template, task=INSTR, input_text=input_text,
                state=state, choices=choices_text(),
                cand_a=verdicts[order[0]], cand_b=verdicts[order[1]])
            work.append((qid, p, order, prompt))

    lock = threading.Lock()
    out = open(args.output, "a", encoding="utf-8")
    finished = [0]

    def one(item):
        qid, p, order, prompt = item
        rec = {"queue_id": qid, "pass": p, "swapped": case_swap(qid),
               "candidate_a_source": order[0],
               "candidate_b_source": order[1],
               "judge_model": args.model,
               "judge_host": args.endpoint.split("//")[-1].split("/")[0],
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            raw = usage = dt = None
            for attempt in range(3):
                try:
                    raw, usage, dt = call_judge(
                        args.endpoint, args.model, prompt, args.api_key,
                        args.temperature, args.timeout, args.json_mode)
                    break
                except requests.Timeout:
                    if attempt == 2:
                        raise
                    time.sleep(10)
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(15)
            v, conf, reason = classify(raw)
            rec.update(raw_response=raw, verdict=v, confidence=conf,
                       reason=reason, latency_s=round(dt, 3),
                       prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"))
        except requests.Timeout:
            rec.update(verdict="TIMEOUT", confidence=0.0, reason="")
        except requests.RequestException as e:
            rec.update(verdict="HTTP_ERROR", confidence=0.0,
                       reason=str(e)[:300])
        return rec

    def emit(rec):
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            finished[0] += 1
            if finished[0] % 50 == 0:
                print(f"  {finished[0]}/{len(work)} calls", flush=True)

    if args.concurrency <= 1:
        for item in work:
            emit(one(item))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(one, it) for it in work]
            for f in as_completed(futs):
                emit(f.result())
    out.close()


# --- analysis ------------------------------------------------------------------

def _real_verdict(rec):
    """Map an A/B verdict back to the real candidate source."""
    if rec["verdict"] == "A":
        return rec["candidate_a_source"]
    if rec["verdict"] == "B":
        return rec["candidate_b_source"]
    return rec["verdict"]          # BOTH / NEITHER / AMBIGUOUS / failures


def analyze(raw_rows, cases, args):
    # Re-classify from raw_response when present: classifier fixes apply
    # retroactively to already-written rows (stored verdict is not trusted).
    for r in raw_rows:
        if r.get("raw_response"):
            v, c, rs = classify(r["raw_response"])
            r["verdict"], r["confidence"], r["reason"] = v, c, rs

    by_case = {}
    for r in raw_rows:
        by_case.setdefault(r["queue_id"], {})[r["pass"]] = r

    n_fail = Counter()
    position_a = position_b = decided = 0
    consistent = inconsistent = 0
    silvers = []
    for case in cases:
        qid = case["queue_id"]
        pr = by_case.get(qid, {})
        r1, r2 = pr.get(1), pr.get(2)
        for r in (r1, r2):
            if not r:
                continue
            if r["verdict"] in ("REFUSAL", "PARSE_FAILURE", "TIMEOUT",
                                "HTTP_ERROR"):
                n_fail[r["verdict"]] += 1
            if r["verdict"] == "A":
                position_a += 1
            elif r["verdict"] == "B":
                position_b += 1
            if r["verdict"] in ("A", "B"):
                decided += 1
        real1 = _real_verdict(r1) if r1 else None
        real2 = _real_verdict(r2) if r2 else None
        is_consistent = (real1 is not None and real1 == real2)
        consistent += bool(is_consistent)
        inconsistent += bool(real1 is not None and not is_consistent)
        minconf = min((r1 or {}).get("confidence", 0.0),
                      (r2 or {}).get("confidence", 0.0))
        fails = [r["verdict"] for r in (r1, r2) if r
                 and r["verdict"] in ("REFUSAL", "PARSE_FAILURE",
                                      "TIMEOUT", "HTTP_ERROR")]
        stable = (is_consistent and not fails
                  and real1 not in ("AMBIGUOUS",)
                  and minconf >= CONF_THRESHOLD)
        silvers.append({
            "queue_id": qid, "sample_id": case["sample_id"],
            "judge_model": (r1 or r2 or {}).get("judge_model"),
            "silver_verdict": real1 if is_consistent else None,
            "stable_silver": stable, "consistent": is_consistent,
            "min_confidence": minconf,
            "pass1_verdict": (r1 or {}).get("verdict"),
            "pass2_verdict": (r2 or {}).get("verdict"),
            "jev_verdict": case["jev_verdict"],
            "jevlite_verdict": case["jevlite_verdict"],
            "needs_review": (not stable) or bool(fails),
            "review_reason": "+".join(
                x for x in [
                    "inconsistent" if not is_consistent else "",
                    "low_confidence" if minconf < CONF_THRESHOLD else "",
                    "ambiguous" if real1 == "AMBIGUOUS" else "",
                    "judge_failure" if fails else ""] if x)})

    pref = Counter(s["silver_verdict"] for s in silvers
                   if s["silver_verdict"])
    pref_stable = Counter(s["silver_verdict"] for s in silvers
                          if s["stable_silver"])

    def acceptable(s, who):
        v = s["silver_verdict"]
        if v == "BOTH":
            return True
        if v in ("NEITHER", "AMBIGUOUS", None):
            return False
        return v == who

    acc = {}
    for who in ("jev", "jevlite"):
        sub = [s for s in silvers if s["silver_verdict"]]
        acc[who] = {
            "acceptable": sum(acceptable(s, who) for s in sub),
            "n_scored": len(sub),
            "acceptable_rate": (sum(acceptable(s, who) for s in sub)
                                / len(sub)) if sub else None,
            "acceptable_rate_stable": (
                sum(acceptable(s, who) for s in silvers
                    if s["stable_silver"])
                / max(1, sum(s["stable_silver"] for s in silvers)))}

    # uncertain analysis
    unc = [s for s in silvers if s["jevlite_verdict"] == "uncertain"]
    unc_judged = [s for s in unc if s["silver_verdict"]]
    unc_summary = {
        "jevlite_uncertain_total": len(unc),
        "uncertain_judged_acceptable": sum(
            s["silver_verdict"] in ("jevlite", "BOTH") for s in unc_judged),
        "uncertain_judged_unacceptable": sum(
            s["silver_verdict"] in ("jev", "NEITHER") for s in unc_judged),
        "uncertain_ambiguous": sum(
            s["silver_verdict"] == "AMBIGUOUS" for s in unc_judged),
        "jev_choice_on_uncertain_cases": dict(Counter(
            s["jev_verdict"] for s in unc)),
    }

    # truncation analysis via token counts (recomputed - items file caps at
    # encode_len so real length is needed for 1025+ buckets)
    tok = {}
    if args.tokens_by_qid:
        tok = args.tokens_by_qid
    buckets = {"<=1024": [], ">1024": [],
               "0-512": [], "513-1024": [], "1025-1536": [], "1537+": []}
    for s in silvers:
        n = tok.get(s["queue_id"])
        if n is None:
            continue
        s["tokens"] = n
        buckets["<=1024" if n <= 1024 else ">1024"].append(s)
        buckets["0-512" if n <= 512 else
                "513-1024" if n <= 1024 else
                "1025-1536" if n <= 1536 else "1537+"].append(s)

    def bucket_stats(sub):
        n = len(sub)
        if not n:
            return {"n": 0}
        scored = [s for s in sub if s["silver_verdict"]]
        return {
            "n": n,
            "jevlite_uncertain_rate": sum(
                s["jevlite_verdict"] == "uncertain" for s in sub) / n,
            "jev_acceptable_rate": (
                sum(acceptable(s, "jev") for s in scored) / len(scored)
                if scored else None),
            "jevlite_acceptable_rate": (
                sum(acceptable(s, "jevlite") for s in scored) / len(scored)
                if scored else None),
            "judge_ambiguous_rate": (
                sum(s["silver_verdict"] == "AMBIGUOUS" for s in scored)
                / len(scored) if scored else None),
            "mean_judge_confidence": float(np.mean(
                [s["min_confidence"] for s in scored])) if scored else None,
        }

    trunc = {k: bucket_stats(v) for k, v in buckets.items()}

    review = [s for s in silvers if s["needs_review"]]

    by_judge = {}
    for s in silvers:
        jm = s.get("judge_model") or "unknown"
        by_judge.setdefault(jm, []).append(s)
    judge_stats = {}
    for jm, sub in by_judge.items():
        scored = [s for s in sub if s["silver_verdict"]]
        judge_stats[jm] = {
            "n": len(sub),
            "stable": sum(s["stable_silver"] for s in sub),
            "consistent": sum(s["consistent"] for s in sub),
            "preferences": dict(Counter(s["silver_verdict"]
                                        for s in scored)),
            "mean_confidence": float(np.mean(
                [s["min_confidence"] for s in scored])) if scored else None,
        }

    lat = [r["latency_s"] for r in raw_rows if r.get("latency_s")]
    ptok = [r["prompt_tokens"] for r in raw_rows if r.get("prompt_tokens")]
    ctok = [r["completion_tokens"]
            for r in raw_rows if r.get("completion_tokens")]
    perf = {
        "n_calls": len(raw_rows),
        "sum_latency_s": round(sum(lat), 1),
        "mean_latency_s": round(float(np.mean(lat)), 2) if lat else None,
        "p50_latency_s": round(float(np.percentile(lat, 50)), 2)
        if lat else None,
        "p95_latency_s": round(float(np.percentile(lat, 95)), 2)
        if lat else None,
        "prompt_tokens_total": sum(ptok) if ptok else None,
        "completion_tokens_total": sum(ctok) if ctok else None,
    }

    second = None
    if args.judge2_file and os.path.exists(args.judge2_file):
        j2 = {}
        for r in iter_jsonl(args.judge2_file):
            if r.get("raw_response"):
                v = classify(r["raw_response"])[0]
            else:
                v = r.get("verdict")
            src = {"A": r.get("candidate_a_source"),
                   "B": r.get("candidate_b_source")}.get(v, v)
            j2.setdefault(r["queue_id"], {})[r["pass"]] = src
        agree = disagree = compared = 0
        for s in silvers:
            pr = j2.get(s["queue_id"])
            if not pr or len(pr) < 2 or not s["silver_verdict"]:
                continue
            v2 = set(pr.values())
            compared += 1
            if v2 == {s["silver_verdict"]}:
                agree += 1
            else:
                disagree += 1
        second = {"file": args.judge2_file, "compared": compared,
                  "agree": agree, "disagree": disagree}

    return {
        "n_cases": len(cases), "n_judged_pairs": len(silvers),
        "swap_consistency": (consistent / max(consistent + inconsistent, 1)),
        "consistent": consistent, "inconsistent": inconsistent,
        "position_A_rate_decided": position_a / max(decided, 1),
        "position_B_rate_decided": position_b / max(decided, 1),
        "failures": dict(n_fail),
        "preference_counts_all": dict(pref),
        "preference_counts_stable": dict(pref_stable),
        "acceptability": acc,
        "uncertain_analysis": unc_summary,
        "truncation_analysis": trunc,
        "by_judge": judge_stats,
        "judge_perf": perf,
        "second_judge": second,
        "stable_silver_n": sum(s["stable_silver"] for s in silvers),
        "review_required_n": len(review),
        "review_reason_dist": dict(Counter(r["review_reason"]
                                           for r in review)),
    }, silvers


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review-file", default="artifacts/jev_human_review.jsonl")
    p.add_argument("--input", nargs="+", default=[],
                   help="original judge queues, to rebuild full state "
                        "(review file's context is truncated to 3000 chars)")
    p.add_argument("--endpoint", default=None,
                   help="OpenAI-compatible base, e.g. http://host:3000/openai")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key-env", default="LLM_JUDGE_API_KEY")
    p.add_argument("--output", default="artifacts/jev_llm_judge_raw.jsonl")
    p.add_argument("--silver-out", default="artifacts/jev_llm_silver_labels.jsonl")
    p.add_argument("--review-required",
                   default="artifacts/jev_llm_review_required.jsonl")
    p.add_argument("--summary", default="artifacts/jev_llm_judge_summary.json")
    p.add_argument("--prompt-file", default=PROMPT_FILE)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--json-mode", action="store_true",
                   help="send response_format=json_object (server-dependent)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel in-flight judge calls (needs server slots)")
    p.add_argument("--shard", default=None,
                   help="I/N - judge only cases with index%%N==I; lets a "
                        "second endpoint process a disjoint shard")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--judge2-file", default=None,
                   help="raw jsonl from a second judge; per-case agreement "
                        "with the primary silvers is reported")
    args = p.parse_args()
    args.api_key = os.environ.get(args.api_key_env)

    cases = list(iter_jsonl(args.review_file))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cases = [c for k, c in enumerate(cases) if k % n == i]
    if args.limit:
        cases = cases[:args.limit]

    cand_by_id = {}
    if args.input:
        cands, imeta = load_inputs(args.input)
        cand_by_id = {c.get("queue_id"): c for c in cands}

    # real token counts for the truncation buckets (items file caps at 1024)
    args.tokens_by_qid = {}
    if args.input:
        try:
            import torch
            ck = torch.load("jevlite.pt", map_location="cpu",
                            weights_only=False)
            measure = make_stats_encoder(ck["encoder"], 1 << 22)
            for c in cands:
                qid = c.get("queue_id")
                if qid in {x["queue_id"] for x in cases}:
                    try:
                        n, _ = measure(jev_state(c), jevlite_questions(c))
                        args.tokens_by_qid[qid] = n
                    except Exception:
                        pass
        except Exception as e:
            print(f"token stats unavailable: {e}", file=sys.stderr)

    if not args.analyze_only:
        if not args.endpoint or not args.model:
            p.error("--endpoint and --model required unless --analyze-only")
        template = open(args.prompt_file, encoding="utf-8").read()
        print(f"judging {len(cases)} cases x {args.passes} passes via "
              f"{args.endpoint} model={args.model}", flush=True)
        run_judge(cases, cand_by_id, template, args)

    raw = list(iter_jsonl(args.output)) if os.path.exists(args.output) else []
    summary, silvers = analyze(raw, cases, args)
    summary["meta"] = {
        "model": args.model, "endpoint_host": (
            args.endpoint.split("//")[-1].split("/")[0]
            if args.endpoint else None),
        "temperature": args.temperature, "passes": args.passes,
        "prompt_file": args.prompt_file,
        "prompt_sha256": sha256_file(args.prompt_file)
        if os.path.exists(args.prompt_file) else None,
        "review_file_sha256": sha256_file(args.review_file)
        if os.path.exists(args.review_file) else None,
        "conf_threshold": CONF_THRESHOLD, "swap_salt": SWAP_SALT,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "host": platform.node(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels_are": "LLM-generated silver labels - NOT human gold"}

    for path, rows in ((args.silver_out, silvers),
                       (args.review_required,
                        [s for s in silvers if s["needs_review"]])):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=1, ensure_ascii=False)

    print(f"\ncases {summary['n_cases']}  swap-consistent "
          f"{summary['consistent']}/{summary['consistent'] + summary['inconsistent']}"
          f"  failures {summary['failures']}")
    print(f"stable silver {summary['stable_silver_n']}  "
          f"review required {summary['review_required_n']}")
    print(f"preferences {summary['preference_counts_all']}")
    for who in ("jev", "jevlite"):
        a = summary["acceptability"][who]
        print(f"{who}: acceptable {a['acceptable_rate']}")
    print(f"uncertain: {summary['uncertain_analysis']}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
