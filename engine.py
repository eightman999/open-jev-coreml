"""One decide() interface, swappable backend.

  engine = DecisionEngine(backend="coreml", compute_units="cpu_and_ne")
  result = engine.decide(state, questions)

backend="torch" delegates to the reference PyTorch implementation in
05_serve.py unchanged - it stays the parity baseline. backend="coreml" runs
encoder+head on Core ML and keeps the dynamic-schema bookkeeping on CPU.

coreml_mode="worker" (the default, the production-safe path) isolates the
ANE runtime in a subprocess - the host-heap corruption documented in
ANE_REPORT.md cannot take down the caller. coreml_mode="inprocess" is a
diagnostic option only; never use it behind a server.
"""
from __future__ import annotations
import importlib.util


def _serve_module():
    """05_serve.py starts with a digit - import it by path like decide.py."""
    spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _coreml_impl(pkg_dir, ckpt, compute_units, coreml_mode, **worker_kwargs):
    if coreml_mode == "worker":
        from coreml_worker import WorkerCoreMLEngine
        return WorkerCoreMLEngine(pkg_dir, ckpt=ckpt,
                                  compute_units=compute_units,
                                  **worker_kwargs), "/worker"
    if coreml_mode == "inprocess":
        from coreml_backend import CoreMLEngine
        return CoreMLEngine(pkg_dir, ckpt=ckpt,
                            compute_units=compute_units), "/inprocess"
    raise ValueError(f"unknown coreml_mode {coreml_mode!r} - "
                     "use 'worker' or 'inprocess'")


class _TorchEngine:
    """Thin wrapper over the reference implementation - do not reimplement it."""
    def __init__(self, ckpt, device, slots=None):
        self._serve = _serve_module()
        self.info = self._serve.load(ckpt, device, slots=slots)
        self.slots = self.info.get("slots", "lead")

    def decide(self, state, questions, calibrated=True):
        return self._serve.decide(state, questions, calibrated)


class DecisionEngine:
    def __init__(self, backend="torch", ckpt="jevlite.pt", device=None,
                 compute_units="cpu_and_ne", pkg_dir="artifacts/coreml",
                 coreml_mode="worker", slots=None, **worker_kwargs):
        self.backend = backend
        if backend == "torch":
            self._impl = _TorchEngine(ckpt, device, slots)
            self.dev = self._impl.info["dev"]
        elif backend == "coreml":
            self._impl, suffix = _coreml_impl(
                pkg_dir, ckpt, compute_units, coreml_mode, **worker_kwargs)
            self.dev = f"coreml:{compute_units}{suffix}"
        else:
            raise ValueError(f"unknown backend {backend!r} - "
                             "use 'torch' or 'coreml'")

    def decide(self, state, questions, calibrated=True):
        """-> {question: {"label", "confidence", "probabilities"}}"""
        return self._impl.decide(state, questions, calibrated)

    def close(self):
        """Shut the worker down cleanly; a no-op for other backends."""
        close = getattr(self._impl, "close", None)
        if close:
            close()
