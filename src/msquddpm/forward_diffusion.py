from __future__ import annotations

import math

import torch

from .channels import depolarizing_channel
from .states import validate_density_matrix
from .trajectory import Trajectory


def noise_schedule(
    steps: int,
    kind: str = "cosine",
    offset: float = 0.001,
    beta_min: float = 1e-4,
    beta_max: float = 1.0,
) -> torch.Tensor:
    """Paper Eqs. (2)-(3); offset/clipping follow the official implementation."""
    if kind == "linear":
        return torch.linspace(1 / steps, 1, steps, dtype=torch.float64)
    t = torch.arange(steps + 1, dtype=torch.float64)
    f = torch.cos(((t / steps + offset) / (1 + offset)) * math.pi / 2).square()
    alpha_bar = f / f[0]
    beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(beta_min, beta_max)
    if kind == "sq_cosine":
        beta = beta.square()
    elif kind != "cosine":
        raise ValueError(f"Unknown schedule: {kind}")
    return beta


class ForwardDiffusion:
    def __init__(self, steps: int, schedule: str = "cosine", offset: float = 0.001):
        self.steps = steps
        self.schedule_name = schedule
        self.betas = noise_schedule(steps, schedule, offset)
        self.offset = offset

    def diffuse(self, rho_0: torch.Tensor, validate: bool = True) -> Trajectory:
        states = {0: rho_0.clone()}
        current = rho_0
        for t, beta in enumerate(self.betas, start=1):
            current = depolarizing_channel(current, beta.to(device=current.device, dtype=current.real.dtype))
            if validate:
                validate_density_matrix(current)
            states[t] = current
        return Trajectory(states, "forward", {"schedule": self.schedule_name, "betas": self.betas.tolist()})
