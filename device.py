"""Shared device selection: cuda > mps > cpu.

Keeping the priority order in one place means the Apple Silicon path never
leaks platform-specific branches into the training and eval scripts - they all
just ask for the best available device. AMP/GradScaler stay CUDA-only at the
call sites that use them; MPS runs the same code in fp32.
"""
import torch


def pick_device(prefer=None):
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
