from __future__ import annotations

import numpy as np
import ot
import torch

from .losses import pairwise_superfidelity
from .states import density_to_bloch, purity


def _cpu128(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.complex128)


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # Diagnostic eigendecomposition: detached CPU complex128 because MPS lacks eigh.
    return torch.linalg.eigvalsh(_cpu128(rho) - _cpu128(sigma)).abs().sum(-1) / 2


def state_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    rho_cpu, sigma_cpu = _cpu128(rho), _cpu128(sigma)
    sqrt_rho = _matrix_sqrt(rho_cpu)
    return torch.linalg.eigvalsh(sqrt_rho @ sigma_cpu @ sqrt_rho).clamp_min(0).sqrt().sum(-1).square()


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.mH) / 2)
    return vectors @ torch.diag_embed(values.clamp_min(0).sqrt()).to(vectors.dtype) @ vectors.mH


def ensemble_wasserstein(a: torch.Tensor, b: torch.Tensor) -> float:
    cost = (1 - pairwise_superfidelity(a, b)).detach().cpu().numpy()
    return float(ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)), cost))


def zero_state_overlap(states: torch.Tensor) -> float:
    return float(states[:, 0, 0].real.mean().detach().cpu())


def nearest_metrics(generated: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    g = pairwise_superfidelity(generated, target)
    nearest = g.argmax(dim=1)
    matched = target[nearest]
    fidelity = state_fidelity(generated, matched).mean().item()
    return {
        "superfidelity": g.max(dim=1).values.mean().item(),
        "fidelity": fidelity,
        "trace_distance": trace_distance(generated, matched).mean().item(),
        "mmd": _mmd(generated, target).item(),
        "wasserstein": ensemble_wasserstein(generated, target),
        "purity_error": (purity(generated).mean() - purity(target).mean()).abs().item(),
        "generated_radius": torch.linalg.vector_norm(density_to_bloch(generated), dim=1).mean().item(),
        "target_radius": torch.linalg.vector_norm(density_to_bloch(target), dim=1).mean().item(),
        "F_data_0": zero_state_overlap(target),
        "F_gen_0": zero_state_overlap(generated),
    }


def _mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return pairwise_superfidelity(a, a).mean() + pairwise_superfidelity(b, b).mean() - 2 * pairwise_superfidelity(a, b).mean()
