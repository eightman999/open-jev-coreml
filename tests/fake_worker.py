"""Fake worker for lifecycle/protocol tests - no Core ML needed.

Speaks the same JSON-lines protocol as coreml_worker.py but decides with a
deterministic hash, so WorkerCoreMLEngine lifecycle tests run on any machine
(Linux CI included, no .mlpackage, no torch):

  python tests/fake_worker.py [--crash-on-call N | --crash-always]
                              [--hang-on-call N]

  --crash-on-call N   os._exit(2) while serving the Nth decide of THIS worker
  --crash-always      os._exit(2) on every decide (deterministic crash)
  --hang-on-call N    sleep forever on the Nth decide (read-timeout path)
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INFO = {"compute_units": "fake", "max_seq": 1024, "max_len": 1024,
        "encode_len": 1024, "temperatures": [1.0, 1.0, 1.0],
        "buckets": [256, 512, 1024]}


def _labels(q):
    """Same label sets typed_schema.iter_labels yields."""
    crit = q.get("criteria")
    if crit is None:
        return ["false", "true"]                    # bare noul
    if isinstance(crit, list):
        return [str(i) for i in range(len(crit))]   # score
    if q["type"] == "noul":
        return [k for k in ("false", "true") if k in crit]
    return sorted(crit)                             # choice


def _decide(state, questions):
    """Deterministic pseudo-decision: hash picks the winner and confidence."""
    out = {}
    for name in sorted(questions):
        q = questions[name]
        labels = _labels(q)
        h = int(hashlib.sha1(
            (json.dumps(state, sort_keys=True, ensure_ascii=False) + name)
            .encode()).hexdigest(), 16)
        win = labels[h % len(labels)]
        conf = 0.40 + (h % 55) / 100.0              # 0.40-0.94
        rest = (1.0 - conf) / max(len(labels) - 1, 1)
        probs = {l: round(rest, 4) for l in labels}
        probs[win] = round(conf, 4)
        out[name] = {"label": win, "confidence": round(conf, 4),
                     "probabilities": probs}
    return out


def main():
    crash_on = hang_on = None
    crash_always = "--crash-always" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--crash-on-call":
            crash_on = int(sys.argv[i + 1])
        if a == "--hang-on-call":
            hang_on = int(sys.argv[i + 1])

    calls = 0
    t_start = time.monotonic()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            if req["op"] == "info":
                res = dict(INFO)
            elif req["op"] == "ping":
                res = {"pong": True}
            elif req["op"] == "stats":
                res = {"calls": calls,
                       "uptime_s": time.monotonic() - t_start,
                       "rss_bytes": 0, "keepalive_len": 0}
            elif req["op"] == "decide":
                calls += 1
                if crash_always or calls == crash_on:
                    os._exit(2)
                if calls == hang_on:
                    time.sleep(600)
                res = {"result": _decide(req["state"], req["questions"]),
                       "label_calls": 1}
            elif req["op"] == "quit":
                return
            else:
                raise ValueError(f"unknown op {req['op']!r}")
            out = {"ok": True, **res}
        except Exception as e:                      # noqa: BLE001 - report it
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
