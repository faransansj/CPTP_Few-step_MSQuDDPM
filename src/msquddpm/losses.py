from __future__ import annotations

import numpy as np
import ot
import torch


def pairwise_superfidelity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    overlap = torch.einsum("aij,bji->ab", a, b).real
    purity_a = torch.einsum("aij,aji->a", a, a).real
    purity_b = torch.einsum("bij,bji->b", b, b).real
    mixed = torch.sqrt((1 - purity_a).clamp_min(0)[:, None] * (1 - purity_b).clamp_min(0)[None, :] + eps)
    return (overlap + mixed).clamp(0, 1)


def superfidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return pairwise_superfidelity(rho[None], sigma[None])[0, 0]


def mmd_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return pairwise_superfidelity(a, a).mean() + pairwise_superfidelity(b, b).mean() - 2 * pairwise_superfidelity(a, b).mean()


def wasserstein_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable uniform OT for equal batches: 1D assignment surrogate is not used.

    POT obtains the exact transport plan on detached costs; gradients flow through
    the selected torch costs (same common practice as the official implementation).
    """
    cost = 1 - pairwise_superfidelity(a, b)
    na, nb = len(a), len(b)
    plan = ot.emd(np.full(na, 1 / na), np.full(nb, 1 / nb), cost.detach().cpu().numpy())
    return (cost * torch.as_tensor(plan, dtype=cost.dtype, device=cost.device)).sum()
