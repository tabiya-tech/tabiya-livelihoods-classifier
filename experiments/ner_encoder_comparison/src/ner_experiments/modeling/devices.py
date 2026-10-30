"""Choosing the device (GPU / CPU) and number format (precision) for a run."""

from typing import Literal

import torch

DeviceName = Literal["cuda", "mps", "cpu"]
PrecisionName = Literal["bf16", "fp16", "fp32"]


def resolve_device(requested: Literal["auto", "cuda", "mps", "cpu"]) -> DeviceName:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch can't see an NVIDIA GPU. On Windows, install the "
            "CUDA build: pip install torch --index-url https://download.pytorch.org/whl/cu126"
        )
    return requested


def resolve_precision(requested: Literal["auto", "bf16", "fp16", "fp32"], device: DeviceName) -> PrecisionName:
    """bf16 on GPUs that support it (the RTX 4060 does), full precision elsewhere."""
    if requested != "auto":
        return requested
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def describe_device(device: DeviceName) -> str:
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        return f"cuda ({properties.name}, {properties.total_memory / 1024**3:.1f} GB)"
    if device == "mps":
        return "mps (Apple GPU)"
    return "cpu (no GPU: training will be very slow)"
