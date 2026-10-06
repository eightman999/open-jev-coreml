"""WorkerCoreMLEngine lifecycle tests - two tiers.

The fake-worker tier runs anywhere (no Core ML, no torch, no packages): it
exercises the spawn/request/respawn/shutdown protocol against
tests/fake_worker.py, which decides by hash. The Core ML tier runs the same
lifecycle against the real ANE worker and skips cleanly without artifacts.

  pytest tests/test_worker.py            # fake tier everywhere
"""
import os
import sys

import pytest

from coreml_worker import WorkerCoreMLEngine, _self_rss_bytes

_FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fake_worker.py")

CKPT = os.environ.get("JEV_CKPT", "jevlite.pt")
PKG_DIR = os.environ.get("JEV_PKG", "artifacts/coreml")

needs_artifacts = pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.isdir(PKG_DIR)
         and any(f.endswith(".mlpackage") for f in os.listdir(PKG_DIR))),
    reason="checkpoint or .mlpackage missing - train + export_coreml.py first")


def fake_engine(*flags, read_timeout=5.0, **kw):
    """WorkerCoreMLEngine pointed at tests/fake_worker.py."""
    return WorkerCoreMLEngine(
        "unused-pkg-dir", ckpt=None, compute_units="fake",
        read_timeout=read_timeout,
        worker_cmd=[sys.executable, _FAKE, *flags], **kw)


@pytest.fixture
def engine():
    e = fake_engine()
    yield e
    e.close()


# --- fake-worker tier: runs anywhere -----------------------------------------

def test_worker_startup(engine):
    """Spawn + info handshake populates the CoreMLEngine metadata surface."""
    assert engine.compute_units == "fake"
    assert engine.max_seq == 1024
    assert engine.temperatures == [1.0, 1.0, 1.0]
    assert engine.buckets == [256, 512, 1024]
    assert engine.ping() is True


def test_decide(engine):
    out = engine.decide("a state", {"q": {"type": "noul",
                                          "instructions": "It rains."}})
    assert out["q"]["label"] in ("false", "true")
    assert 0.0 <= out["q"]["confidence"] <= 1.0
    assert abs(sum(out["q"]["probabilities"].values()) - 1.0) < 0.05


def test_dynamic_schema(engine):
    """Arbitrary question names/types are answered without any fixed head."""
    out = engine.decide(
        {"anything": 1},
        {"soup": {"type": "choice", "instructions": "Rate the soup.",
                  "criteria": {"gourmet": "great", "hazard": "call 911"}},
         "ok": {"type": "noul", "instructions": "All is well."},
         "sev": {"type": "score", "instructions": "How bad?",
                 "criteria": ["low", "med", "high"]}})
    assert out["soup"]["label"] in ("gourmet", "hazard")
    assert out["ok"]["label"] in ("false", "true")
    assert out["sev"]["label"] in ("0", "1", "2")


def test_multiple_questions_one_pass(engine):
    out = engine.decide("s", {"a": {"type": "noul", "instructions": "A?"},
                              "b": {"type": "noul", "instructions": "B?"}})
    assert engine.last_label_calls == 1     # one encoder pass, not one per Q
    assert set(out) == {"a", "b"}


def test_clean_shutdown(engine):
    engine.close()
    engine._p.wait(timeout=5)
    assert engine._p.poll() == 0          # quit op -> clean exit, not a kill


def test_crash_respawns_once():
    """decide #2 kills the worker; parent respawns and the retry succeeds."""
    e = fake_engine("--crash-on-call", "2")
    try:
        assert "a" in e.decide("s", {"a": {"type": "noul",
                                           "instructions": "A?"}})
        assert "a" in e.decide("s", {"a": {"type": "noul",
                                           "instructions": "A?"}})
        assert e.restarts == 1
        assert e.calls == 2
    finally:
        e.close()


def test_deterministic_crash_fails_after_one_retry():
    """A worker that crashes every decide yields a clear RuntimeError."""
    e = fake_engine("--crash-always")
    try:
        with pytest.raises(RuntimeError, match="worker died"):
            e.decide("s", {"a": {"type": "noul", "instructions": "A?"}})
        assert e.restarts == 1              # retried once, not forever
    finally:
        e.close()


def test_read_timeout():
    """A hung worker is killed, respawned once, then RuntimeError."""
    e = fake_engine("--hang-on-call", "1", read_timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="worker died"):
            e.decide("s", {"a": {"type": "noul", "instructions": "A?"}})
        assert e.restarts == 1
    finally:
        e.close()


def test_recycle_bounds_worker_lifetime():
    """recycle_every respawns the worker on a schedule - keepalive bounded."""
    e = fake_engine(recycle_every=3)
    try:
        for _ in range(7):
            e.decide("s", {"a": {"type": "noul", "instructions": "A?"}})
        assert e.recycles == 2              # after call 3 and call 6
        assert e.restarts == 0              # recycling is planned, not a crash
        assert e.calls == 7
    finally:
        e.close()


def test_worker_stats_op(engine):
    st = engine.worker_stats()
    assert st["calls"] == 0 and st["uptime_s"] >= 0
    engine.decide("s", {"a": {"type": "noul", "instructions": "A?"}})
    assert engine.worker_stats()["calls"] == 1


def test_parent_memory_bounded(engine):
    """300 proxied decisions must not grow the PARENT's RSS linearly."""
    q = {"a": {"type": "noul", "instructions": "A?"}}
    engine.decide("warm", q)                 # settle allocator noise
    before = _self_rss_bytes()
    for _ in range(300):
        engine.decide("x" * 2000, q)
    after = _self_rss_bytes()
    assert after - before < 64 * 1024 * 1024, (
        f"parent RSS grew {(after - before) / 1e6:.0f} MB over 300 requests")


# --- Core ML tier: needs real packages ----------------------------------------

@needs_artifacts
def test_real_worker_startup_and_decide():
    e = WorkerCoreMLEngine(PKG_DIR, ckpt=CKPT, compute_units="cpu_and_ne")
    try:
        assert e.compute_units == "cpu_and_ne" and e.max_seq >= 1024
        from typed_schema import Question
        out = e.decide("refund request for a duplicate charge",
                       {"r": Question.noul("This is about a refund.")})
        assert out["r"]["label"] in ("true", "false")
    finally:
        e.close()


@needs_artifacts
def test_real_worker_kill_respawns_once():
    """SIGKILL on the live worker: next decide respawns and succeeds."""
    e = WorkerCoreMLEngine(PKG_DIR, ckpt=CKPT, compute_units="cpu_only")
    try:
        from typed_schema import Question
        q = {"r": Question.noul("This is about a refund.")}
        e.decide("refund me", q)
        e._p.kill()                          # simulate the ANE heap crash
        e._p.wait(timeout=10)
        out = e.decide("refund me", q)       # EOF -> respawn -> retry -> ok
        assert out["r"]["label"] in ("true", "false")
        assert e.restarts == 1
    finally:
        e.close()
