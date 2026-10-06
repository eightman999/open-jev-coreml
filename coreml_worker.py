"""Production worker-process isolation for the Core ML / ANE backend.

  DecisionEngine / 05_serve --backend coreml
        |  JSON lines over stdin/stdout
        v
  python coreml_worker.py <pkg_dir> <ckpt|-> <compute_units>
        |
        v
  CoreMLEngine -> Core ML / Apple Neural Engine

Why a subprocess: on this machine the ANE runtime can corrupt the host heap
when a bridged predict output is deallocated on a background dispatch thread
(objc_destructInstance -> _PyObject_Free, then unrelated native calls fault;
see ANE_REPORT.md "failed experiments"). A disposable child bounds the blast
radius - a worker crash is an EOF on a pipe, not a SIGSEGV inside a serving
process, and CoreMLEngine's _keepalive list (which must retain every predict
output to avoid that dealloc) is bounded by the worker's lifetime.

Protocol - one JSON object per line, request in / response out:
  {"op": "info"}     -> engine metadata (compute_units, buckets, temps, ...)
  {"op": "decide", "state": ..., "questions": ..., "calibrated": bool}
                     -> {"result": {...}, "label_calls": int}
  {"op": "stats"}    -> {"calls": int, "uptime_s": float, "rss_bytes": int}
  {"op": "ping"}     -> {"pong": true}
  {"op": "quit"}     -> worker exits

Every response carries "ok": true or {"ok": false, "error": "..."}.

WorkerCoreMLEngine is the parent-side client. It exposes the same practical
surface as CoreMLEngine (decide + metadata attributes), spawns the worker
eagerly at init, respawns once and retries a request once on worker death or
read timeout, and can recycle the worker every `recycle_every` decisions so
the retained predict outputs stay bounded in a long-running service.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from collections import deque

READ_TIMEOUT_S = 120.0      # a hung worker is as dead as a crashed one
RECYCLE_EVERY = 10000       # decisions per worker lifetime before recycling
_CLOSE_TIMEOUT_S = 10.0
_STDERR_LINES = 200

_WORKER_PY = os.path.abspath(__file__)


def _self_rss_bytes() -> int:
    """Current RSS - /proc on Linux, `ps` elsewhere (macOS)."""
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


def worker_main(argv) -> int:
    """Child-process entrypoint: CoreMLEngine behind the JSON-lines protocol."""
    pkg_dir, ckpt, cu = argv[0], argv[1], argv[2]
    seq_lens = None
    if len(argv) > 3:
        seq_lens = [int(x) for x in argv[3].split(",") if x]
    from coreml_backend import CoreMLEngine
    eng = CoreMLEngine(pkg_dir, ckpt=None if ckpt == "-" else ckpt,
                       compute_units=cu, seq_lens=seq_lens)

    calls = 0
    t_start = time.monotonic()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            out = {"ok": False, "error": f"bad request json: {e}"}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            continue
        try:
            if req["op"] == "info":
                res = {"compute_units": eng.compute_units,
                       "encoder": eng.encoder_name,
                       "max_seq": eng.max_seq, "max_len": eng.max_len,
                       "encode_len": eng.encode_len,
                       "temperatures": eng.temperatures,
                       "slots": eng.slots,
                       "buckets": sorted(eng._models)}
            elif req["op"] == "ping":
                res = {"pong": True}
            elif req["op"] == "stats":
                res = {"calls": calls,
                       "uptime_s": time.monotonic() - t_start,
                       "rss_bytes": _self_rss_bytes(),
                       "keepalive_len": len(eng._keepalive)}
            elif req["op"] == "decide":
                calls_before = calls
                orig = eng.label_logits

                def counting(enc):
                    nonlocal calls
                    calls += 1
                    return orig(enc)

                eng.label_logits = counting
                try:
                    res = {"result": eng.decide(
                               req["state"], req["questions"],
                               req.get("calibrated", True)),
                           "label_calls": calls - calls_before}
                finally:
                    eng.label_logits = orig
            elif req["op"] == "quit":
                return 0
            else:
                raise ValueError(f"unknown op {req['op']!r}")
            out = {"ok": True, **res}
        except Exception as e:                      # noqa: BLE001 - report it
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


class WorkerCoreMLEngine:
    """decide() proxy for a CoreMLEngine living in a worker subprocess.

    Same practical interface as CoreMLEngine - construction takes the same
    (packages, ckpt, compute_units, seq_lens) arguments and metadata arrives
    via the worker's info response.

    Lifecycle:
      - worker dies or a read times out -> kill/reap, respawn, retry once;
        a second failure raises RuntimeError (no respawn loop).
      - `recycle_every` decisions per worker bounds CoreMLEngine._keepalive,
        which must retain every predict output (releasing them is what trips
        the ANE heap bug). Recycling is transparent and lazy - it happens at
        the start of the next decide(), so a quiet service pays nothing.
      - close() asks the worker to quit, then kills if it does not.

    `worker_cmd` overrides the spawn command - tests use it to point at a
    fake worker that needs no Core ML runtime.
    """

    def __init__(self, packages="artifacts/coreml", ckpt=None,
                 compute_units="cpu_and_ne", seq_lens=None,
                 read_timeout=READ_TIMEOUT_S, recycle_every=RECYCLE_EVERY,
                 worker_cmd=None):
        self._args = (packages, ckpt, compute_units, seq_lens)
        self.read_timeout = read_timeout
        self.recycle_every = recycle_every
        self._worker_cmd = list(worker_cmd) if worker_cmd else None
        self.calls = 0            # successful decide() calls, all workers
        self.restarts = 0         # crash/timeout respawns (unplanned)
        self.recycles = 0         # planned respawns (keepalive bound)
        self.last_label_calls = 0
        self._calls_this_worker = 0
        self._stderr = deque(maxlen=_STDERR_LINES)
        self._spawn()
        info = self._call({"op": "info"})
        self.compute_units = info["compute_units"]
        self.encoder_name = info.get("encoder")
        self.max_seq = info["max_seq"]
        self.max_len = info["max_len"]
        self.encode_len = info["encode_len"]
        self.temperatures = info["temperatures"]
        self.slots = info.get("slots", "lead")
        self.buckets = info["buckets"]

    # -- process management -------------------------------------------------

    def _spawn(self):
        old = getattr(self, "_p", None)
        if old is not None and old.poll() is None:
            old.kill()                      # hung predecessor, not a dead one
            try:
                old.wait(timeout=_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
        pkg_dir, ckpt, cu, seq_lens = self._args
        cmd = self._worker_cmd
        if cmd is None:
            cmd = [sys.executable, _WORKER_PY, pkg_dir,
                   ckpt if ckpt else "-", cu]
            if seq_lens:
                cmd.append(",".join(str(s) for s in seq_lens))
        self._p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(_WORKER_PY))
        self._rbuf = b""
        self._calls_this_worker = 0
        # stderr is drained continuously into a bounded buffer: it never
        # blocks the worker on a full pipe, and it is there for diagnostics
        # if the worker dies.
        self._stderr.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._p.stderr,), daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self, stream):
        try:
            for line in stream:
                self._stderr.append(
                    line.decode(errors="replace").rstrip())
        except (ValueError, OSError):
            pass

    def _stderr_tail(self, n=5):
        return "; ".join(list(self._stderr)[-n:]) or "(no stderr)"

    def _read_response_line(self):
        """One '\n'-terminated line, bounded by read_timeout. None on
        timeout/EOF. Uses os.read + an own buffer: buffered text IO can
        hold a second response that select() then never reports."""
        while b"\n" not in self._rbuf:
            ready = select.select([self._p.stdout], [], [],
                                  self.read_timeout)[0]
            if not ready:
                return None
            chunk = os.read(self._p.stdout.fileno(), 1 << 16)
            if not chunk:
                return None
            self._rbuf += chunk
        line, self._rbuf = self._rbuf.split(b"\n", 1)
        return line

    def _call(self, req, _retries=1):
        assert self._p.stdin and self._p.stdout
        try:
            self._p.stdin.write(
                (json.dumps(req, ensure_ascii=False) + "\n").encode())
            self._p.stdin.flush()
            line = self._read_response_line()
        except (BrokenPipeError, ValueError, OSError):
            line = None
        if line is None:
            if req["op"] == "quit":         # already dead - nothing to quit
                return {"ok": True}
            if _retries <= 0:
                raise RuntimeError(
                    f"coreml worker died or timed out on {req['op']!r} "
                    f"(exit {self._p.poll()}, stderr: {self._stderr_tail()})")
            self.restarts += 1
            self._spawn()                   # respawn and retry once
            return self._call(req, _retries - 1)
        res = json.loads(line)
        if not res["ok"]:
            raise RuntimeError(f"coreml worker: {res['error']}")
        return res

    # -- engine surface ------------------------------------------------------

    def decide(self, state, questions, calibrated=True):
        """Same shape as CoreMLEngine.decide() / 05_serve.decide()."""
        if self.recycle_every and self._calls_this_worker >= self.recycle_every:
            # Retained predict outputs bound the ANE heap bug; recycling the
            # worker frees them with the process instead of deallocating in
            # place, which is the unsafe operation.
            self.recycles += 1
            self._recycle()
        res = self._call({"op": "decide", "state": state,
                          "questions": questions, "calibrated": calibrated})
        self.calls += 1
        self._calls_this_worker += 1
        self.last_label_calls = res["label_calls"]
        return res["result"]

    def ping(self):
        return self._call({"op": "ping"})["pong"]

    def worker_stats(self):
        """Worker-side counters incl. RSS - for benchmarks and leak checks."""
        return self._call({"op": "stats"})

    def _recycle(self):
        try:
            self._call({"op": "quit"})
            self._p.wait(timeout=_CLOSE_TIMEOUT_S)
        except Exception:
            try:
                self._p.kill()
            except Exception:
                pass
        self._spawn()

    def close(self):
        try:
            self._call({"op": "quit"})
        except Exception:
            pass
        try:
            self._p.wait(timeout=_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._p.kill()
            try:
                self._p.wait(timeout=_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:]))
