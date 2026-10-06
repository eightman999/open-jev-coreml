"""Core ML / Apple Neural Engine backend for JevLite.

The .mlpackage computes token_logits [1, S] - the score head evaluated at every
position. Everything that makes the schema dynamic stays here on CPU: encoding
the question text, gathering the <<l>> marker positions, dividing by the
per-type temperature, and the per-question softmax. Adding or renaming a
question therefore needs no new weights and no new export.

  from coreml_backend import CoreMLEngine
  eng = CoreMLEngine("artifacts/coreml", ckpt="jevlite.pt",
                     compute_units="cpu_and_ne")
  eng.decide(state, questions)
"""
from __future__ import annotations
import json
import os
import time

import numpy as np

SEQ_LENS = (256, 512, 1024)          # default export buckets, largest last


def _compute_unit(name):
    import coremltools as ct
    return {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }[name]


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class CoreMLEngine:
    """Fixed-shape Core ML inference behind the same decide() interface.

    packages:  a .mlpackage path, or a directory of jevlite-<seq>.mlpackage
               bundles (smallest bundle >= input length wins).
    ckpt:      optional .pt checkpoint; supplies encoder name, max_len and the
               calibrated per-type temperatures. Package metadata is used when
               no checkpoint is given.
    """

    def __init__(self, packages="artifacts/coreml", ckpt=None,
                 compute_units="cpu_and_ne", seq_lens=None):
        import coremltools as ct
        from transformers import AutoTokenizer
        from td_data import add_markers

        self.compute_units = compute_units
        cu = _compute_unit(compute_units)

        # -- locate packages -------------------------------------------------
        meta = {}
        if os.path.isdir(packages):
            found = {}
            for f in os.listdir(packages):
                if f.endswith(".mlpackage"):
                    try:
                        s = int(f.rsplit("-", 1)[1].split(".")[0])
                    except (ValueError, IndexError):
                        continue
                    found[s] = os.path.join(packages, f)
            if not found:
                raise FileNotFoundError(
                    f"no jevlite-*.mlpackage under {packages} - run "
                    "export_coreml.py first")
            want = seq_lens or sorted(found)
            self._pkg = {s: found[s] for s in want if s in found}
            if not self._pkg:
                raise FileNotFoundError(
                    f"no jevlite-*.mlpackage for seq_lens {sorted(seq_lens)} "
                    f"under {packages} - available: {sorted(found)}")
            manifest = os.path.join(packages, "manifest.json")
            if os.path.exists(manifest):
                meta = json.load(open(manifest))
        else:
            self._pkg = {}

        # -- load models eagerly so load() time is honest --------------------
        self._models, self.load_time_s = {}, {}
        for s, path in sorted(self._pkg.items()):
            t0 = time.perf_counter()
            self._models[s] = ct.models.MLModel(path, compute_units=cu)
            self.load_time_s[s] = time.perf_counter() - t0
        if not self._models:
            # a single package path: load it and read seq_len off its metadata
            t0 = time.perf_counter()
            only = ct.models.MLModel(packages, compute_units=cu)
            self.load_time_s[0] = time.perf_counter() - t0
            s = int(only.user_defined_metadata.get("seq_len")
                    or (seq_lens[0] if seq_lens else 0))
            if not s:
                raise ValueError(f"cannot infer seq_len of {packages}")
            self._models[s] = only
            self.load_time_s = {s: self.load_time_s.pop(0)}

        # -- model metadata: checkpoint wins, package/manifest fills gaps ----
        ck = None
        if ckpt and os.path.exists(ckpt):
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck = ck or {}
        first = self._models[sorted(self._models)[0]]
        self.encoder_name = (ck.get("encoder") or meta.get("encoder")
                             or first.user_defined_metadata.get("encoder"))
        self.max_len = int(ck.get("max_len") or meta.get("max_len")
                           or first.user_defined_metadata.get("max_len")
                           or max(self._models))
        # per-type temperature; index order matches typed_schema.TYPES. The
        # manifest stores it as a JSON array, the package metadata as a JSON
        # string - accept either shape.
        def _temps(v):
            return v if v is None or isinstance(v, list) else json.loads(v)
        temps = (ck.get("temperatures") or _temps(meta.get("temperatures"))
                 or _temps(first.user_defined_metadata.get("temperatures")))
        self.temperatures = list(temps) if temps else [1.0, 1.0, 1.0]
        # Label serialization the checkpoint was trained under; pre-slots
        # checkpoints get the current default ("lead" - the pos0 fix).
        self.slots = ck.get("slots") or meta.get("slots") or "lead"

        self.tok = AutoTokenizer.from_pretrained(self.encoder_name)
        _, self.qid, self.lid = add_markers(self.tok)
        self.pad_id = self.tok.pad_token_id
        # Predict outputs are ObjC-bridged objects; releasing them lets Core ML
        # destroy a graph containing PyObject* on a background dispatch thread,
        # which corrupts the heap (EXC_BAD_ACCESS in objc_destructInstance ->
        # _PyObject_Free, later faults in unrelated native calls). Retaining the
        # outputs for the engine's lifetime sidesteps the dealloc entirely -
        # each is only a few KB.
        self._keepalive = []

        self.max_seq = max(self._models)
        # Never feed the encoder a longer buffer than the largest package.
        self.encode_len = min(self.max_len, self.max_seq)

    # -- inference ----------------------------------------------------------

    def _bucket(self, n):
        for s in sorted(self._models):
            if s >= n:
                return s
        return self.max_seq

    def token_logits(self, input_ids):
        """Pad to the smallest covering bucket and run the package. -> [S]

        Over-length input raises rather than truncating: the questions live at
        the tail of the sequence, so chopping it would silently drop them. The
        state-absorbs-truncation policy lives in td_data.encode and is applied
        at encode_len before we get here.
        """
        L = len(input_ids)
        if L > self.max_seq:
            raise ValueError(
                f"encoded input is {L} tokens but the largest package is "
                f"{self.max_seq} - export a bigger bucket "
                f"(export_coreml.py --seq-lens {self.max_seq * 2})")
        S = self._bucket(L)
        ids = np.full((1, S), self.pad_id, dtype=np.int32)
        att = np.zeros((1, S), dtype=np.int32)
        ids[0, :L] = input_ids[:L]
        att[0, :L] = 1
        out = self._models[S].predict(
            {"input_ids": ids, "attention_mask": att})
        self._keepalive.append(out)
        return np.array(out["token_logits"], dtype=np.float32)[0]

    def label_logits(self, enc):
        """token_logits gathered at the encoded <<l>> positions. -> [M]"""
        tl = self.token_logits(enc["input_ids"])
        return tl[np.asarray(enc["label_pos"])]

    def decide(self, state, questions, calibrated=True):
        """Same shape as the PyTorch decide(): label/confidence/probabilities."""
        from td_data import encode

        row = {"id": "live", "state": state, "questions": questions, "gold": {}}
        enc = encode(row, self.tok, self.encode_len, self.qid, self.lid,
                     with_gold=False, slots=self.slots)
        return self.decide_enc(enc, calibrated)

    def decide_enc(self, enc, calibrated=True):
        """decide() on a pre-encoded row - the path the parity harness uses."""
        lg = self.label_logits(enc)
        out, i = {}, 0
        for gi, qname in enumerate(enc["qnames"]):
            labels = enc["labels"][gi]
            n = len(labels)
            t = self.temperatures[enc["qtype"][gi]] if calibrated else 1.0
            probs = _softmax(lg[i:i + n] / t)
            dist = dict(zip(labels, (round(float(v), 4) for v in probs)))
            top = max(dist, key=dist.get)
            out[qname] = {"label": top, "confidence": dist[top],
                          "probabilities": dist}
            i += n
        return out
