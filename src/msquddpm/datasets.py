from __future__ import annotations

import math

import torch

from .precision import precision_for
from .states import validate_density_matrix


def depolarize(states: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(states.shape[-1], dtype=states.dtype, device=states.device) / states.shape[-1]
    return (1 - q[..., None, None]) * states + q[..., None, None] * eye


def clustered_states(
    size: int,
    seed: int = 42,
    epsilon: float = 0.08,
    q_max: float = 0.01,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    device = torch.device(device); precision = precision_for(device)
    # Generate on CPU because torch.Generator(device='mps') is unsupported, then
    # transfer once in the device-selected precision.
    generator = torch.Generator().manual_seed(seed)
    c = torch.randn(size, 2, generator=generator, dtype=precision.real)
    psi = torch.stack((torch.ones(size, dtype=precision.real), epsilon * torch.complex(c[:, 0], c[:, 1])), dim=1).to(precision.complex)
    psi /= torch.linalg.vector_norm(psi, dim=1, keepdim=True)
    pure = torch.einsum("bi,bj->bij", psi, psi.conj())
    q = torch.rand(size, generator=generator, dtype=precision.real) * q_max
    states = depolarize(pure, q).to(device)
    validate_density_matrix(states)
    return states


def circular_states(
    size: int,
    seed: int = 42,
    q_max: float = 0.04,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    device = torch.device(device); precision = precision_for(device)
    generator = torch.Generator().manual_seed(seed)
    theta = torch.rand(size, generator=generator, dtype=precision.real) * (2 * math.pi)
    psi = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2)), dim=1).to(precision.complex)
    pure = torch.einsum("bi,bj->bij", psi, psi.conj())
    q = torch.rand(size, generator=generator, dtype=precision.real) * q_max
    states = depolarize(pure, q).to(device)
    validate_density_matrix(states)
    return states


def make_dataset(name: str, size: int, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    if name == "clustered":
        return clustered_states(size, seed=seed, device=device)
    if name == "circular":
        return circular_states(size, seed=seed, device=device)
    raise ValueError(f"Unknown dataset: {name}")
