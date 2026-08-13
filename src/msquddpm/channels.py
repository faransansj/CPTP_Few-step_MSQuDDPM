from __future__ import annotations

import torch


def depolarizing_channel(rho: torch.Tensor, q: float | torch.Tensor) -> torch.Tensor:
    """Paper Eq. (1), applied once to the full density matrix per timestep."""
    probability = torch.as_tensor(q, dtype=rho.real.dtype, device=rho.device)
    eye = torch.eye(rho.shape[-1], dtype=rho.dtype, device=rho.device) / rho.shape[-1]
    return (1 - probability[..., None, None]) * rho + probability[..., None, None] * eye
