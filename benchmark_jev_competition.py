"""Jev vs local JevLite competition benchmark on the novllm judge workload.

Reproduces the historical judging semantics exactly: the same state string,
the same choice question (noise / content / uncertain), the same context
truncation (6000 chars) that tools/judge_jev.py sent to the Jev API
(api.typesafe.ai systemone, model jev-1.13.0). What differs is only the
engine - and the fact that JevLite encodes at max_len=1024 tokens, so very
long contexts lose tail characters that Jev saw. Every truncated item is
flagged in the report; this is a *task-transfer* caveat, not hidden.

  # agreement on cached verdicts - zero API cost
  python benchmark_jev_competition.py \
      --input  judge/longspan/queue_*.resplit.jsonl \
      --jev-results judge/longspan/verdicts_*.resplit.jev.jsonl \
      --backend coreml:cpu_and_ne --out artifacts/jev_competition.json

Levels: 1) paired agreement vs historical Jev (NOT accuracy - Jev is not
ground truth), 2) stratified human-review sample + scoring once a --gold
file is filled in, 3) systems throughput/latency/memory.

--items-in re-analyses a previous run's --items-out without inference.
--live-jev queries the real API; it is opt-in, never default, and needs
TYPESAFE_API_KEY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

# --- historical task definition (verbatim from novllm tools/judge_jev.py) ---

JEV_BASE_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_MAX_CTX_CHARS = 6000

INSTR = (
    "あなたは小説テキストのコーパス浄化判定器です。state は小説レコードの末尾付近の"
    "抜粋で、「候補区間」は除去候補としてマークされた範囲です。確認事項に基づき"
    "候補区間が何であるかを一つ選んでください。本文の要約・引用・再生成は不要です。"
)

CRITERIA = {
    "noise": "除去すべき取得ノイズ: サイト案内文・ランキング/ブックマーク/評価の呼びかけ・"
             "作者の宣伝や前書き後書き・奥付メタデータ・ナビゲーション文字列など",
    "content": "物語・本文の一部として残すべき文章",
    "uncertain": "情報不足や境界が曖昧で判断できないもの",
}

VERDICTS = ["content", "noise", "uncertain"]      # sorted, matches iter_labels
ROUTING_THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def jev_state(cand: dict) -> str:
    """The exact `state` string judge_jev.py sent to the Jev API."""
    import ast
    ctx = (cand.get("context") or "")[:JEV_MAX_CTX_CHARS]
    span = cand.get("span") or {}
    if isinstance(span, str):           # some queues store a repr, not a dict
        try:
            span = ast.literal_eval(span)
        except (ValueError, SyntaxError):
            span = {}
    return (
        f"対象レコード: {cand.get('record_id', '')} "
        f"(category={cand.get('category', '')})\n"
        f"確認事項: {cand.get('question', '')}\n"
        f"候補区間: 文字 {span.get('start', '?')}〜{span.get('end', '?')}\n\n"
        f"{ctx}"
    )


def jevlite_questions(cand: dict) -> dict:
    """The exact `questions` payload judge_jev.py sent - a 3-way choice."""
    return {"judge": {"type": "choice",
                      "instructions": INSTR + "確認事項: "
                                    + str(cand.get("question", "")),
                      "criteria": CRITERIA}}


# --- IO ----------------------------------------------------------------------

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


def load_inputs(paths):
    items, meta = [], []
    for p in paths:
        rows = [r for r in iter_jsonl(p)]
        meta.append({"path": p, "n": len(rows), "sha256": sha256_file(p)})
        items.extend(rows)
    return items, meta


def load_verdicts(paths):
    """queue_id -> verdict row. Duplicate queue_ids keep the first verdict."""
    out, meta, dups = {}, [], 0
    for p in paths:
        n = 0
        for r in iter_jsonl(p):
            n += 1
            qid = r.get("queue_id")
            if qid and qid not in out:
                out[qid] = r
            elif qid:
                dups += 1
        meta.append({"path": p, "n": n, "sha256": sha256_file(p)})
    return out, meta, dups


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine()}
    try:
        out = subprocess.run(["sysctl", "-n", "hw.model"],
                             capture_output=True, text=True, timeout=5)
        info["hw_model"] = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _rss_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


# --- engine adapters -----------------------------------------------------------

def make_engine(backend, args):
    """-> (decide_fn, meta, close_fn). decide_fn(state, questions) -> result."""
    kind, _, opt = backend.partition(":")
    if kind == "coreml":
        if args.coreml_mode == "worker":
            from coreml_worker import WorkerCoreMLEngine
            eng = WorkerCoreMLEngine(
                args.pkg_dir, ckpt=args.ckpt, compute_units=opt or "cpu_and_ne",
                read_timeout=args.read_timeout,
                recycle_every=args.recycle_every)
            meta = {"engine": "WorkerCoreMLEngine", "compute_units": opt,
                    "encoder": eng.encoder_name, "encode_len": eng.encode_len}
        else:
            from coreml_backend import CoreMLEngine
            eng = CoreMLEngine(args.pkg_dir, ckpt=args.ckpt,
                               compute_units=opt or "cpu_and_ne")
            meta = {"engine": "CoreMLEngine(inprocess)", "compute_units": opt,
                    "encoder": eng.encoder_name, "encode_len": eng.encode_len}

        def decide(state, questions):
            return eng.decide(state, questions)

        def stats():
            s = {"restarts": getattr(eng, "restarts", 0),
                 "recycles": getattr(eng, "recycles", 0)}
            try:
                s["worker"] = eng.worker_stats()
            except Exception:
                pass
            return s

        return decide, meta, eng.close, stats

    if kind == "torch":
        import importlib.util
        spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        e = S.load(args.ckpt, device=opt or None)
        meta = {"engine": "JevLite(torch)", "device": e["dev"],
                "encode_len": e["max_len"],
                "encoder": e["model"].encoder_name}

        def decide(state, questions):
            return S.decide(state, questions)

        return decide, meta, lambda: None, lambda: {}

    if kind == "live-jev":
        if not args.live_jev:
            raise SystemExit("--backend live-jev requires --live-jev "
                             "(real API calls cost money - explicit opt-in)")
        import requests
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("TYPESAFE_API_KEY not set")

        def decide(state, questions):
            payload = {"model": JEV_MODEL, "state": state,
                       "questions": questions}
            r = requests.post(JEV_BASE_URL, json=payload,
                              headers={"Authorization": f"Bearer {key}"},
                              timeout=180)
            r.raise_for_status()
            ans = r.json()["answers"]["judge"]
            return {"judge": {"label": ans.get("choice", "uncertain"),
                              "confidence": float(ans.get("confidence", 0.0)),
                              "probabilities": ans.get("probabilities") or {}}}

        return decide, {"engine": "jev-api", "model": JEV_MODEL,
                        "encode_len": None}, lambda: None, lambda: {}

    raise SystemExit(f"unknown --backend {backend!r} "
                     "(coreml:<unit> | torch:<device> | live-jev)")


def make_stats_encoder(encoder_name, encode_len, slots="lead"):
    """Parent-side tokenizer for per-item token/truncation stats.

    Only used for measurement; inference encoding happens inside the engine.
    """
    from transformers import AutoTokenizer
    from td_data import add_markers, encode

    tok = AutoTokenizer.from_pretrained(encoder_name)
    _, qid, lid = add_markers(tok)

    def measure(state, questions):
        row = {"id": "bench", "state": state, "questions": questions,
               "gold": {}}
        enc = encode(row, tok, encode_len, qid, lid, with_gold=False,
                     slots=slots)
        n = len(enc["input_ids"])
        truncated = False
        if n >= encode_len:
            full = encode(row, tok, 1 << 22, qid, lid, with_gold=False,
                          slots=slots)
            truncated = len(full["input_ids"]) > n
        return n, truncated

    return measure


# --- level 2: stratified human-review sample ------------------------------------

def _conf_bin(c):
    for i in range(len(CONF_BINS) - 1):
        if CONF_BINS[i] <= c < CONF_BINS[i + 1]:
            return i
    return len(CONF_BINS) - 2


def stratified_sample(paired, n, seed):
    """Deterministic stratified sample over (agree x confbin x len-tercile).

    Disagreements are the informative cases and are guaranteed representation:
    every disagreement stratum contributes before agreement strata fill the
    rest proportionally. Fixed seed -> identical sample on rerun.
    """
    lens = sorted(it["chars"] for it in paired)
    t1 = lens[len(lens) // 3]
    t2 = lens[2 * len(lens) // 3]

    def len_bucket(it):
        L = it["chars"]
        return 0 if L <= t1 else (1 if L <= t2 else 2)

    strata = {}
    for it in paired:
        key = ("agree" if it["agree"] else "disagree",
               _conf_bin(it["jevlite_confidence"]), len_bucket(it))
        strata.setdefault(key, []).append(it)

    rng = random.Random(seed)
    picked = []
    # disagreements first - they carry the review signal
    dis = [k for k in strata if k[0] == "disagree"]
    dis_items = [it for k in dis for it in strata[k]]
    rng.shuffle(dis_items)
    take_dis = min(len(dis_items), max(n // 3, min(len(dis_items), n)))
    picked.extend(dis_items[:take_dis])

    rest_keys = [k for k in strata if k[0] == "agree"]
    budget = n - len(picked)
    total = sum(len(strata[k]) for k in rest_keys)
    for k in sorted(rest_keys):
        stratum = strata[k]
        rng.shuffle(stratum)
        quota = max(1, round(budget * len(stratum) / total)) if total else 0
        picked.extend(stratum[:quota])
    picked = picked[:n]
    picked.sort(key=lambda it: it["queue_id"])
    return picked


def candidate_text(cand):
    """Best-effort extraction of the marked span for the review file."""
    if cand.get("text"):
        return cand["text"]
    ctx = cand.get("context") or ""
    span = cand.get("span") or {}
    s, e = span.get("start"), span.get("end")
    if isinstance(s, int) and isinstance(e, int):
        ctx_start = max(0, s - 250)     # corpus_cleanse._make_candidate window
        return ctx[s - ctx_start:e - ctx_start]
    return ""


def write_sample(sample, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, it in enumerate(sample):
            f.write(json.dumps({
                "sample_id": f"s{i:04d}",
                "queue_id": it["queue_id"], "record_id": it["record_id"],
                "rule": it.get("rule"),
                "candidate_text": candidate_text(it["cand"]),
                "context": (it["cand"].get("context") or "")[:3000],
                "jev_verdict": it["jev_verdict"],
                "jev_confidence": it["jev_confidence"],
                "jevlite_verdict": it["jevlite_label"],
                "jevlite_confidence": it["jevlite_confidence"],
                "agree": it["agree"],
                "human_label": "", "notes": "",
            }, ensure_ascii=False) + "\n")


# --- analysis --------------------------------------------------------------------

def _hist(vals):
    out = [0] * (len(CONF_BINS) - 1)
    for v in vals:
        out[_conf_bin(v)] += 1
    labels = [f"{CONF_BINS[i]:.2f}-{CONF_BINS[i+1]:.2f}" for i in range(len(out))]
    return dict(zip(labels, out))


def _summary(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"n": len(vals), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90))}


def _tvd(p, q):
    """Total variation distance between two prob dicts."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0)))
                     for k in keys)


def _brier_ece(items, get_probs, get_label, get_conf):
    """Brier + 10-bin ECE for one system against human gold."""
    gold_items = [it for it in items if it.get("human_label") in VERDICTS]
    if not gold_items:
        return None
    brier = sum(sum((get_probs(it).get(v, 0.0)
                     - (1.0 if it["human_label"] == v else 0.0)) ** 2
                    for v in VERDICTS) for it in gold_items) / len(gold_items)
    bins = {}
    for it in gold_items:
        c = float(get_conf(it))
        bins.setdefault(min(int(c * 10), 9), []).append(
            float(get_label(it) == it["human_label"]))
    ece = sum(len(v) / len(gold_items) * abs(
        sum(v) / len(v) - (b / 10 + 0.05)) for b, v in bins.items())
    return {"brier": brier, "ece10": ece, "n": len(gold_items),
            "bins": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {
                "n": len(v), "acc": sum(v) / len(v)} for b, v in bins.items()}}


def _prf(conf, labels):
    """Macro precision/recall/F1 from a confusion dict {gold: {pred: n}}."""
    per = {}
    for lab in labels:
        tp = conf.get(lab, {}).get(lab, 0)
        fp = sum(conf.get(g, {}).get(lab, 0) for g in labels) - tp
        fn = sum(conf.get(lab, {}).values()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {"precision": p, "recall": r,
                    "f1": 2 * p * r / (p + r) if p + r else 0.0}
    macro = {m: sum(per[l][m] for l in labels) / len(labels)
             for m in ("precision", "recall", "f1")}
    return {"per_label": per, "macro": macro}


def gold_metrics(items):
    """Both systems scored independently against filled human_label rows."""
    gold = [it for it in items if it.get("human_label") in VERDICTS]
    if not gold:
        return None
    out = {}
    for name in ("jev", "jevlite"):
        lab = f"{name}_verdict" if name == "jev" else "jevlite_label"
        pred = [it[lab] for it in gold]
        truth = [it["human_label"] for it in gold]
        conf = {}
        for t, p in zip(truth, pred):
            conf.setdefault(t, {}).setdefault(p, 0)
            conf[t][p] += 1
        acc = sum(t == p for t, p in zip(truth, pred)) / len(gold)
        rec = {"n": len(gold), "accuracy": acc, "confusion": conf,
               **_prf(conf, VERDICTS)}
        if name == "jevlite":
            rec["calibration"] = _brier_ece(
                items, lambda it: it["jevlite_probs"],
                lambda it: it["jevlite_label"],
                lambda it: it["jevlite_confidence"])
        else:
            rec["calibration"] = _brier_ece(
                items, lambda it: it["jev_probs"],
                lambda it: it["jev_verdict"],
                lambda it: it["jev_confidence"])
        out[name] = rec
    return out


def analyze(items, verdicts, args, cand_by_id=None):
    """Pure function of recorded items + verdicts -> report dict."""
    paired = [it for it in items
              if it["queue_id"] in verdicts and "jevlite_label" in it]
    unpaired = len(items) - len(paired)

    for it in paired:
        v = verdicts[it["queue_id"]]
        it["jev_verdict"] = v.get("verdict")
        it["jev_confidence"] = float(v.get("confidence") or 0.0)
        it["jev_probs"] = v.get("probabilities") or {}
        it["agree"] = it["jev_verdict"] == it["jevlite_label"]
        if cand_by_id and it["queue_id"] in cand_by_id:
            it["cand"] = cand_by_id[it["queue_id"]]

    agree = [it for it in paired if it["agree"]]
    dis = [it for it in paired if not it["agree"]]

    confusion = {jv: {lv: 0 for lv in VERDICTS} for jv in VERDICTS}
    for it in paired:
        confusion[it["jev_verdict"]][it["jevlite_label"]] += 1

    tvds = [_tvd(it["jev_probs"], it["jevlite_probs"]) for it in paired
            if it["jev_probs"] and it["jevlite_probs"]]

    level1 = {
        "paired_n": len(paired), "unpaired_or_failed": unpaired,
        "agreement": len(agree) / len(paired) if paired else None,
        "disagreement_n": len(dis),
        "confusion_jev_x_jevlite": confusion,
        "jev_verdict_dist": {v: sum(1 for it in paired
                                    if it["jev_verdict"] == v)
                             for v in VERDICTS},
        "jevlite_verdict_dist": {v: sum(1 for it in paired
                                        if it["jevlite_label"] == v)
                                 for v in VERDICTS},
        "confidence_agree": _summary([it["jevlite_confidence"]
                                      for it in agree]) if agree else None,
        "confidence_disagree": _summary([it["jevlite_confidence"]
                                         for it in dis]) if dis else None,
        "confidence_hist_agree": _hist([it["jevlite_confidence"]
                                        for it in agree]),
        "confidence_hist_disagree": _hist([it["jevlite_confidence"]
                                           for it in dis]),
        "tvd_probs_mean": float(np.mean(tvds)) if tvds else None,
        "truncated_inputs": sum(1 for it in paired if it.get("truncated")),
        # agreement reliability: P(jev agreement) per confidence bin - NOT
        # calibration (Jev is a reference system, not ground truth).
        "agreement_reliability": {
            f"{CONF_BINS[i]:.2f}-{CONF_BINS[i+1]:.2f}": {
                "n": sum(1 for it in paired
                         if _conf_bin(it["jevlite_confidence"]) == i),
                "agree_frac": (lambda b: sum(
                    it["agree"] for it in b) / len(b) if b else None)(
                    [it for it in paired
                     if _conf_bin(it["jevlite_confidence"]) == i])}
            for i in range(len(CONF_BINS) - 1)},
    }

    routing = []
    for thr in ROUTING_THRESHOLDS:
        local = [it for it in paired if it["jevlite_confidence"] >= thr]
        routing.append({
            "threshold": thr,
            "local_frac": len(local) / len(paired) if paired else 0,
            "agreement_local": (sum(it["agree"] for it in local) / len(local)
                                if local else None),
            "escalated_frac": 1 - (len(local) / len(paired) if paired else 0)})

    report = {"level1_paired_agreement": level1, "routing_table": routing}

    # level 2: stratified human-review sample (+ gold scoring if filled)
    if args.sample_n and paired and cand_by_id:
        sample = stratified_sample(
            [it for it in paired if "cand" in it], args.sample_n, args.seed)
        if args.sample_out:
            write_sample(sample, args.sample_out)
        report["level2_human_sample"] = {
            "n": len(sample), "seed": args.seed, "out": args.sample_out,
            "disagreements_included": sum(1 for it in sample
                                          if not it["agree"]),
            "sample_queue_ids": [it["queue_id"] for it in sample]}
    if args.gold and os.path.exists(args.gold):
        gold_map = {r["queue_id"]: r["human_label"]
                    for r in iter_jsonl(args.gold)
                    if r.get("human_label")}
        for it in items:
            if it["queue_id"] in gold_map:
                it["human_label"] = gold_map[it["queue_id"]]
        report["level2_gold"] = gold_metrics(items)
    return report


# --- level 3: systems run -------------------------------------------------------

def run_inference(items, decide, args):
    """Returns (per-item records, systems stats). Order = input order."""
    results, failures = [], 0
    consecutive_failures = 0
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    for i, cand in enumerate(items):
        state = jev_state(cand)
        questions = jevlite_questions(cand)
        t = time.perf_counter()
        try:
            res = decide(state, questions)
            dt = (time.perf_counter() - t) * 1000
            consecutive_failures = 0
            a = res["judge"]
            rec = {"queue_id": cand.get("queue_id"),
                   "record_id": cand.get("record_id"),
                   "rule": cand.get("rule"),
                   "jevlite_label": a["label"],
                   "jevlite_confidence": a["confidence"],
                   "jevlite_probs": a["probabilities"],
                   "latency_ms": dt, "chars": len(state),
                   "warmup": i < args.warmup}
        except Exception as e:                       # noqa: BLE001 - count it
            dt = (time.perf_counter() - t) * 1000
            failures += 1
            consecutive_failures += 1
            rec = {"queue_id": cand.get("queue_id"),
                   "record_id": cand.get("record_id"),
                   "rule": cand.get("rule"), "error": str(e),
                   "latency_ms": dt, "chars": len(state),
                   "warmup": i < args.warmup}
            if consecutive_failures > args.max_consecutive_failures:
                results.append(rec)
                print(f"\naborting: {consecutive_failures} consecutive "
                      "failures - the backend is not serving",
                      file=sys.stderr)
                break
        results.append(rec)
        if (i + 1) % 200 == 0:
            done = i + 1
            el = time.perf_counter() - t0
            print(f"  {done}/{len(items)}  {el:.0f}s "
                  f"({done / el:.1f}/s, {failures} failed)", flush=True)
    wall = time.perf_counter() - t0
    return results, {"wall_s": wall, "failures": failures,
                     "rss_before": rss_before, "rss_after": _rss_bytes()}


def systems_stats(results, run, meta, load_s, engine_stats):
    ok = [r for r in results if "jevlite_label" in r]
    steady = [r["latency_ms"] for r in ok if not r["warmup"]]
    cold = [r["latency_ms"] for r in ok if r["warmup"]]
    a = np.asarray(steady, dtype=np.float64)
    tokens = sum(r.get("tokens") or 0 for r in ok)
    return {
        "backend": meta, "load_s": load_s, "n_total": len(results),
        "n_ok": len(ok), "n_failed": run["failures"],
        "wall_s": run["wall_s"],
        "decisions_per_s": len(ok) / run["wall_s"] if run["wall_s"] else 0,
        "chars_per_s": sum(r["chars"] for r in ok) / run["wall_s"]
        if run["wall_s"] else 0,
        "tokens_per_s": tokens / run["wall_s"] if run["wall_s"] else 0,
        "latency_ms": {"p50": float(np.percentile(a, 50)) if len(a) else None,
                       "p90": float(np.percentile(a, 90)) if len(a) else None,
                       "p95": float(np.percentile(a, 95)) if len(a) else None,
                       "p99": float(np.percentile(a, 99)) if len(a) else None,
                       "mean": float(a.mean()) if len(a) else None,
                       "max": float(a.max()) if len(a) else None},
        "cold_warmup_latency_ms": _summary(cold) if cold else None,
        "worker_restarts": engine_stats.get("restarts", 0),
        "worker_recycles": engine_stats.get("recycles", 0),
        "worker_stats": engine_stats.get("worker"),
        "rss_parent_before": run["rss_before"],
        "rss_parent_after": run["rss_after"],
        "rss_parent_delta_mb": (run["rss_after"] - run["rss_before"]) / 1e6,
    }


# --- main ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=[],
                   help="judge queue JSONL(s), concatenated in order")
    p.add_argument("--jev-results", nargs="+", default=[],
                   help="historical Jev verdict JSONL(s), paired by queue_id")
    p.add_argument("--backend", default="coreml:cpu_and_ne",
                   help="coreml:<unit> | torch:<device> | live-jev")
    p.add_argument("--coreml-mode", default="worker",
                   choices=["worker", "inprocess"])
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--pkg-dir", default="artifacts/coreml")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=50,
                   help="first N items excluded from latency percentiles")
    p.add_argument("--read-timeout", type=float, default=120.0)
    p.add_argument("--recycle-every", type=int, default=10000)
    p.add_argument("--max-consecutive-failures", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-n", type=int, default=400)
    p.add_argument("--sample-out", default="artifacts/jev_human_review.jsonl")
    p.add_argument("--items-out", default=None,
                   help="per-item results JSONL (default artifacts/"
                        "jev_items_<backend>.jsonl)")
    p.add_argument("--items-in", default=None,
                   help="skip inference; analyse a previous --items-out file")
    p.add_argument("--disagreements-out",
                   default="artifacts/jev_disagreements.jsonl")
    p.add_argument("--gold", default=None,
                   help="filled human-review JSONL -> level-2 scoring")
    p.add_argument("--jev-source", default=None,
                   help="path to judge_jev.py for provenance hash")
    p.add_argument("--live-jev", action="store_true",
                   help="allow real Jev API calls (never default)")
    p.add_argument("--out", default="artifacts/jev_competition.json")
    args = p.parse_args()

    verdicts, vmeta, vdups = ({}, [], 0)
    if args.jev_results:
        verdicts, vmeta, vdups = load_verdicts(args.jev_results)

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "backend": args.backend, "coreml_mode": args.coreml_mode,
            "machine": machine_info(), "seed": args.seed,
            "warmup": args.warmup, "batch_size": 1,
            "timing_includes_tokenization": True,
            "inputs": None, "jev_results": vmeta,
            "jev_duplicate_queue_ids": vdups,
            "task_semantics": {
                "source": "novllm tools/judge_jev.py (Jev systemone payload)",
                "source_sha256": (sha256_file(args.jev_source)
                                  if args.jev_source and
                                  os.path.exists(args.jev_source) else None),
                "instructions": INSTR, "criteria": CRITERIA,
                "state_template": "対象レコード: {id} (category={cat})\\n"
                                  "確認事項: {q}\\n候補区間: 文字 {s}〜{e}\\n\\n"
                                  "{ctx[:6000]}",
                "jevlite_mapping": "Question.choice(INSTR + '確認事項: ' + "
                                   "question, CRITERIA) - identical payload",
                "known_differences": "Jev saw ctx[:6000] chars; JevLite "
                                     "encodes <=1024 tokens so long states "
                                     "lose tail characters (counted in "
                                     "truncated_inputs)"}}

    cand_by_id = {}
    if args.input:
        cands, imeta = load_inputs(args.input)
        meta["inputs"] = imeta
        cand_by_id = {c.get("queue_id"): c for c in cands}

    if args.items_in:
        results = list(iter_jsonl(args.items_in))
        meta["items_in"] = {"path": args.items_in, "n": len(results),
                            "sha256": sha256_file(args.items_in)}
    else:
        if not args.input:
            p.error("--input or --items-in required")
        if args.limit:
            cands = cands[:args.limit]
        print(f"{len(cands)} inputs, {len(verdicts)} historical verdicts",
              flush=True)

        t0 = time.perf_counter()
        decide, emeta, close_fn, estats = make_engine(args.backend, args)
        load_s = time.perf_counter() - t0
        print(f"engine loaded in {load_s:.1f}s: {emeta}", flush=True)

        measure = None
        if emeta.get("encoder") and emeta.get("encode_len"):
            try:
                measure = make_stats_encoder(emeta["encoder"],
                                             emeta["encode_len"])
            except Exception as e:
                print(f"stats tokenizer unavailable: {e}", file=sys.stderr)

        results, run = run_inference(cands, decide, args)
        engine_stats = estats()
        close_fn()

        # per-item token/truncation stats (post-hoc; inference already timed)
        if measure:
            print("measuring token/truncation stats...", flush=True)
            for cand, r in zip(cands, results):
                try:
                    n, tr = measure(jev_state(cand), jevlite_questions(cand))
                    r["tokens"], r["truncated"] = n, tr
                except Exception:
                    r["tokens"], r["truncated"] = None, None

        items_out = args.items_out or \
            f"artifacts/jev_items_{args.backend.replace(':', '_')}.jsonl"
        os.makedirs(os.path.dirname(items_out) or ".", exist_ok=True)
        with open(items_out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {items_out} ({len(results)} rows)", flush=True)

        meta["engine"] = emeta
        meta["items_out"] = items_out
        report_sys = systems_stats(results, run, emeta, load_s, engine_stats)
    report = analyze(results, verdicts, args, cand_by_id)
    report["meta"] = meta
    if not args.items_in:
        report["level3_systems"] = report_sys

    if args.disagreements_out:
        dis = [r for r in results
               if r.get("jev_verdict") and r.get("jevlite_label")
               and r["jev_verdict"] != r["jevlite_label"]]
        os.makedirs(os.path.dirname(args.disagreements_out) or ".",
                    exist_ok=True)
        with open(args.disagreements_out, "w", encoding="utf-8") as f:
            for r in dis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)

    l1 = report["level1_paired_agreement"]
    print(f"\npaired {l1['paired_n']}  agreement "
          f"{(l1['agreement'] or 0) * 100:.1f}%  "
          f"disagreements {l1['disagreement_n']}")
    if not args.items_in:
        s = report["level3_systems"]
        print(f"{s['n_ok']} ok in {s['wall_s']:.0f}s "
              f"({s['decisions_per_s']:.1f}/s)  "
              f"p50 {s['latency_ms']['p50']:.0f}ms "
              f"p95 {s['latency_ms']['p95']:.0f}ms  "
              f"restarts {s['worker_restarts']}  "
              f"RSS +{s['rss_parent_delta_mb']:.0f}MB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
