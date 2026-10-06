"""Export JevLite to Core ML: encoder + score head as a fixed-shape ML Program.

The exported graph computes `token_logits [B, L]` - the Linear(d, 1) head
applied at EVERY token position. That is the same math as the reference
`JevLite.logits()` gather-then-Linear, but it keeps the dynamic label gather
out of the Core ML program, so the question/label schema stays data.

  python export_coreml.py --ckpt jevlite.pt --seq-lens 256 512 1024

Output: one .mlpackage per sequence length, fp16, ML Program, in --outdir.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import coremltools as ct

from model import JevLite, require_ckpt

# --- missing coremltools op -------------------------------------------------
# transformers builds its composed attention masks with `tensor.new_ones()` /
# `new_zeros()` (masking_utils.and_masks/or_masks). coremltools 9 registers
# new_zeros but not new_ones, so any model whose mask path composes functions -
# ModernBERT's sliding-window layers do - dies at conversion. Registering the
# op here is additive: it changes nothing for models that never emit it.
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.type_mapping import builtin_to_string
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op, _TORCH_OPS_REGISTRY,
)
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs, _get_kwinputs, _cast_to,
)
from coremltools.converters.mil.frontend.torch.utils import NUM_TO_DTYPE_STRING


@register_torch_op
def new_ones(context, node):
    """tensor.new_ones(size, dtype=None, ...) -> fill(shape, 1) in `dtype`."""
    inputs = _get_inputs(context, node)
    shape = inputs[1]
    if isinstance(shape, list):
        shape = (mb.concat(values=shape, axis=0) if shape
                 else mb.const(val=np.zeros(0, dtype=np.int32)))
    elif not types.is_int(shape.dtype):
        shape = mb.cast(x=shape, dtype="int32")
    res = mb.fill(shape=shape, value=1.0)

    dtype = inputs[2] if len(inputs) > 2 else None
    dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
    if dtype is not None and getattr(dtype, "val", None) is not None:
        dtype_str = NUM_TO_DTYPE_STRING[dtype.val]
    else:
        dtype_str = builtin_to_string(inputs[0].dtype)
    context.add(_cast_to(res, dtype_str, node.name), node.name)


# EXIR spells `a & b` / `a | b` between tensors as dunder ops. register_torch_op
# rejects any name ending in "_" as "inplace" (even though is_inplace_op exempts
# dunders), so these have to go straight into the registry.
def _dunder_and(context, node):
    inputs = _get_inputs(context, node)
    x, y = inputs[0], inputs[1]
    if types.is_bool(x.dtype) and types.is_bool(y.dtype):
        context.add(mb.logical_and(x=x, y=y, name=node.name))
    else:
        context.add(mb.bitwise_and(x=x, y=y, name=node.name))


def _dunder_or(context, node):
    inputs = _get_inputs(context, node)
    x, y = inputs[0], inputs[1]
    if types.is_bool(x.dtype) and types.is_bool(y.dtype):
        context.add(mb.logical_or(x=x, y=y, name=node.name))
    else:
        context.add(mb.bitwise_or(x=x, y=y, name=node.name))


_TORCH_OPS_REGISTRY.set_func_by_name(_dunder_and, "__and__")
_TORCH_OPS_REGISTRY.set_func_by_name(_dunder_or, "__or__")


class TokenLogitsWrapper(torch.nn.Module):
    """Encoder + score head over every position. -> token_logits [B, L]"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m.token_logits(input_ids, attention_mask)


def export_one(wrapper, seq_len, out_path, precision="fp16",
               min_macos=15, verbose=True):
    """Convert at one fixed (1, seq_len) int32 input and save the package."""
    ids = torch.zeros(1, seq_len, dtype=torch.int32)
    att = torch.ones(1, seq_len, dtype=torch.int32)

    kwargs = dict(
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=ids.shape, dtype=np.int32, name="input_ids"),
            ct.TensorType(shape=att.shape, dtype=np.int32, name="attention_mask"),
        ],
        outputs=[ct.TensorType(name="token_logits")],
        compute_precision=(ct.precision.FLOAT16 if precision == "fp16"
                           else ct.precision.FLOAT32),
        minimum_deployment_target=getattr(ct.target, f"macOS{min_macos}"),
    )

    t0 = time.time()
    try:
        ep = torch.export.export(wrapper, (ids, att)).run_decompositions({})
        ml = ct.convert(ep, **kwargs)
        how = "torch.export"
    except Exception as e:
        if verbose:
            print(f"  torch.export path failed ({type(e).__name__}: "
                  f"{str(e)[:200]}) - falling back to jit.trace")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (ids, att), strict=False)
        ml = ct.convert(traced, **kwargs)
        how = "jit.trace"
    if verbose:
        print(f"  converted via {how} in {time.time() - t0:.0f}s -> {out_path}")

    ml.save(out_path)
    return ml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1024],
                 help="one .mlpackage per length; 1024 matches --max-len")
    p.add_argument("--outdir", default="artifacts/coreml")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                 help="fp16 is what makes the Neural Engine worth using; "
                      "fp32 is the parity fallback")
    p.add_argument("--min-macos", type=int, default=26,
                   help="DO NOT lower for fp16: macOS15-targeted fp16 kernels "
                        "corrupt this model (max logit err 5.5 vs 0.04 at 26). "
                        "See ANE_REPORT.md 'failed experiments'.")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    wrapper = TokenLogitsWrapper(model).eval()

    os.makedirs(a.outdir, exist_ok=True)
    manifest = {"ckpt": a.ckpt, "encoder": ck["encoder"],
                "max_len": ck["max_len"], "precision": a.precision,
                "min_macos": a.min_macos,
                "temperatures": ck.get("temperatures"),
                "seq_lens": a.seq_lens, "packages": {}}

    for L in a.seq_lens:
        out = os.path.join(a.outdir, f"jevlite-{L}.mlpackage")
        print(f"exporting seq_len={L}")
        ml = export_one(wrapper, L, out, a.precision, a.min_macos)

        # Self-describing packages: the backend can run with no checkpoint.
        ml.user_defined_metadata.update({
            "encoder": ck["encoder"],
            "max_len": str(ck["max_len"]),
            "seq_len": str(L),
            "temperatures": json.dumps(ck.get("temperatures")),
            "score_head": "token_logits",
        })
        ml.save(out)

        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(out) for f in fs)
        manifest["packages"][str(L)] = {
            "path": out, "bytes": size,
        }
        print(f"  {size/1e6:.0f} MB")

    mf = os.path.join(a.outdir, "manifest.json")
    json.dump(manifest, open(mf, "w"), indent=1)
    print(f"\nwrote {mf}")
    print("verify: python verify_coreml.py --ckpt", a.ckpt)


if __name__ == "__main__":
    main()
