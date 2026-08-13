from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    real: torch.dtype
    complex: torch.dtype
    validation_atol: float


def precision_for(device: str | torch.device) -> Precision:
    """MPS supports float32/complex64; CPU and CUDA retain research precision."""
    device = torch.device(device)
    return (
        Precision(torch.float32, torch.complex64, 2e-5)
        if device.type == "mps"
        else Precision(torch.float64, torch.complex128, 1e-7)
    )


def complex_dtype(real: torch.dtype) -> torch.dtype:
    return torch.complex64 if real == torch.float32 else torch.complex128
