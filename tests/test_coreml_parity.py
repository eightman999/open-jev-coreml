"""Core ML backend tests.

The reformulation test runs anywhere - it is pure torch math, no downloads.
Everything touching a .mlpackage skips cleanly when the artifacts have not been
exported yet (or when a checkpoint is absent), so the suite is safe to run on a
fresh clone:

  pytest tests/test_coreml_parity.py            # skips Core ML parts if absent
  JEV_CKPT=jevlite.pt JEV_PKG=artifacts/coreml pytest tests/
"""
import os

import numpy as np
import pytest
import torch

# Imported up front on purpose: transformers 5.x lazily re-executes its own
# __init__ (direct_transformers_import -> rglob over models/) the first time
# AutoModel is instantiated, and doing that after the ANE runtime has run in
# this process has been observed to segfault. Importing before any Core ML
# predict keeps the fragile path off the post-ANE heap.
import transformers.modeling_layers  # noqa: F401
from model import grouped_softmax  # noqa: F401  (pulls model + transformers)

CKPT = os.environ.get("JEV_CKPT", "jevlite.pt")
PKG_DIR = os.environ.get("JEV_PKG", "artifacts/coreml")

needs_artifacts = pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.isdir(PKG_DIR)
         and any(f.endswith(".mlpackage") for f in os.listdir(PKG_DIR))),
    reason="checkpoint or .mlpackage missing - train + export_coreml.py first")


# --- pure-math reformulation ------------------------------------------------

def test_linear_before_after_gather_equivalence():
    """Linear(d,1) is per-token: gather->Linear == Linear-all->gather."""
    torch.manual_seed(0)
    B, L, D, M = 3, 47, 768, 9
    h = torch.randn(B, L, D)
    score = torch.nn.Linear(D, 1)
    label_pos = torch.randint(0, L, (B, M))
    d = h.size(-1)

    ref = score(h.gather(1, label_pos.unsqueeze(-1).expand(-1, -1, d))) \
        .squeeze(-1)
    got = score(h).squeeze(-1).gather(1, label_pos)
    # same math; GEMM tiling makes bitwise equality too much to ask
    torch.testing.assert_close(ref, got, rtol=1e-5, atol=1e-6)


def test_grouped_softmax_matches_reference():
    """The numpy per-question softmax equals model.grouped_softmax."""
    from coreml_backend import _softmax
    torch.manual_seed(1)
    lg = torch.randn(1, 7)
    group = torch.tensor([[0, 0, 1, 1, 1, -1, -1]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    ref = grouped_softmax(lg, group, mask, 2)[0]
    got = np.concatenate([_softmax(lg[0, :2].numpy()),
                          _softmax(lg[0, 2:5].numpy()),
                          np.zeros(2)])
    assert np.allclose(ref.numpy(), got, atol=1e-6)


# --- Core ML engine ---------------------------------------------------------
#
# The engine runs in the production worker subprocess (coreml_worker.py) -
# the same isolation the serving path uses, so the suite exercises what
# ships. A dead worker surfaces as a clean RuntimeError instead of a
# mid-suite SIGSEGV from the ANE heap issue documented in ANE_REPORT.md.

from coreml_worker import WorkerCoreMLEngine


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("coremltools")
    e = WorkerCoreMLEngine(PKG_DIR, CKPT, "cpu_and_ne")
    yield e
    e.close()


@pytest.fixture(scope="module")
def demo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s.DEMO_STATE, s.DEMO_QUESTIONS


@needs_artifacts
def test_cpu_and_ne_backend_loads(engine):
    assert engine.compute_units == "cpu_and_ne"
    assert engine.max_seq >= 1024


@needs_artifacts
def test_noul(engine):
    from typed_schema import Question
    out = engine.decide("the deploy wiped the test database twice",
                        {"bad": Question.noul("The incident caused data loss.")})
    assert out["bad"]["label"] in ("true", "false")
    assert 0.0 <= out["bad"]["confidence"] <= 1.0
    assert abs(sum(out["bad"]["probabilities"].values()) - 1.0) < 0.01


@needs_artifacts
def test_choice(engine):
    from typed_schema import Question
    out = engine.decide("refund request for a duplicate charge",
                        {"route": Question.choice(
                            "Where should this go?",
                            {"billing": "money questions",
                             "technical": "broken things",
                             "other": "anything else"})})
    assert out["route"]["label"] in ("billing", "technical", "other")


@needs_artifacts
def test_score(engine):
    from typed_schema import Question
    out = engine.decide("agent deleted 3 accounts it was told to preserve",
                        {"severity": Question.score(
                            "How severe?", ["low", "medium", "high"])})
    assert out["severity"]["label"] in ("0", "1", "2")


@needs_artifacts
def test_multiple_questions_one_pass(engine, demo):
    state, questions = demo
    out = engine.decide(state, questions)
    assert engine.last_label_calls == 1     # one encoder pass, not one per Q
    assert set(out) == set(questions)


@needs_artifacts
def test_1024_input(engine):
    from typed_schema import Question
    big = {"log": ["x" * 4000] * 40}       # far over 1024 tokens once encoded
    out = engine.decide(big, {"ok": Question.noul("Everything is fine.")})
    assert out["ok"]["label"] in ("true", "false")


@needs_artifacts
def test_dynamic_schema(engine):
    """Questions never seen in training still get answered - schema is data."""
    from typed_schema import Question
    out = engine.decide(
        {"borscht": {"beets": 3, "rating": "excellent"}},
        {"soup_quality": Question.choice(
            "Rate the soup.", {"gourmet": "great", "edible": "ok",
                               "hazard": "call poison control"}),
         "is_borscht": Question.noul("The dish is borscht.")})
    assert out["soup_quality"]["label"] in ("gourmet", "edible", "hazard")
    assert out["is_borscht"]["label"] in ("true", "false")


@needs_artifacts
def test_pytorch_coreml_decision_parity(engine):
    """End-to-end decision agreement vs the PyTorch reference on real rows."""
    import importlib.util
    from td_data import load_split
    from model import require_ckpt

    spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    s.load(CKPT, device="cpu")
    rows = load_split("test", limit=8)
    ref = [s.decide(r["state"], r["questions"]) for r in rows]

    ck = require_ckpt(CKPT)
    if engine.max_seq < ck["max_len"]:
        pytest.skip("packages do not cover the checkpoint max_len - the "
                    "engines would encode different-length inputs")

    agree = tot = 0
    conf_errs = []
    for row, r in zip(rows, ref):
        got = engine.decide(row["state"], row["questions"])
        for q in r:
            agree += int(r[q]["label"] == got[q]["label"])
            conf_errs.append(abs(r[q]["confidence"] - got[q]["confidence"]))
            tot += 1
    assert agree / tot >= 0.95, f"decision agreement {agree}/{tot}"
    assert np.mean(conf_errs) <= 0.02


@needs_artifacts
def test_package_metadata_only_load():
    """ckpt=None: encoder/max_len/temperatures come from manifest + package.

    The manifest stores temperatures as a JSON array while package metadata
    stores the same value as a JSON string - both must parse.
    """
    from typed_schema import Question
    e = WorkerCoreMLEngine(PKG_DIR, ckpt=None, compute_units="cpu_only")
    try:
        assert len(e.temperatures) == 3 and all(t > 0 for t in e.temperatures)
        assert e.max_len > 0 and e.max_seq >= 1024
        out = e.decide("refund request for a duplicate charge",
                       {"r": Question.noul("This is about a refund.")})
        assert out["r"]["label"] in ("true", "false")
    finally:
        e.close()


@needs_artifacts
def test_seq_lens_filter_rejects_missing_bucket():
    """Asking for an unexported bucket fails clearly - before any model load,
    not with a cryptic MLModel-on-a-directory error."""
    pytest.importorskip("coremltools")
    from coreml_backend import CoreMLEngine
    with pytest.raises(FileNotFoundError, match="seq_lens"):
        CoreMLEngine(PKG_DIR, ckpt=CKPT, compute_units="cpu_only",
                     seq_lens=[8192])
