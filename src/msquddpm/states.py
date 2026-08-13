from __future__ import annotations

import warnings

import numpy as np
import torch

from .precision import complex_dtype, precision_for

_PAULI_VALUES = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]


def _pauli(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_PAULI_VALUES, dtype=dtype, device=device)


def bloch_to_density(vector: torch.Tensor | np.ndarray) -> torch.Tensor:
    raw = torch.as_tensor(vector)
    device = raw.device
    precision = precision_for(device)
    r = raw.to(dtype=precision.real)
    eye = torch.eye(2, dtype=precision.complex, device=device)
    return (eye + torch.einsum("...k,kij->...ij", r.to(precision.complex), _pauli(device, precision.complex))) / 2


def density_to_bloch(rho: torch.Tensor | np.ndarray) -> torch.Tensor:
    state = torch.as_tensor(rho)
    dtype = state.dtype if state.is_complex() else complex_dtype(state.dtype)
    state = state.to(dtype=dtype)
    return torch.einsum("...ij,kji->...k", state, _pauli(state.device, dtype)).real


def purity(rho: torch.Tensor | np.ndarray) -> torch.Tensor:
    state = torch.as_tensor(rho)
    return torch.einsum("...ij,...ji->...", state, state).real


def project_density_matrix(rho: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Diagnostic repair on CPU; not used in differentiable model execution."""
    original_device, original_dtype = rho.device, rho.dtype
    hermitian = ((rho.detach().cpu().to(torch.complex128) + rho.detach().cpu().to(torch.complex128).mH) / 2)
    values, vectors = torch.linalg.eigh(hermitian)
    values = values.clamp_min(eps)
    projected = vectors @ torch.diag_embed(values).to(vectors.dtype) @ vectors.mH
    projected /= projected.diagonal(dim1=-2, dim2=-1).sum(-1).real[..., None, None]
    return projected.to(device=original_device, dtype=original_dtype)


def validate_density_matrix(
    rho: torch.Tensor | np.ndarray,
    atol: float | None = None,
    raise_error: bool = True,
) -> dict[str, float | bool]:
    raw = torch.as_tensor(rho)
    if raw.ndim < 2 or raw.shape[-1] != raw.shape[-2]:
        raise ValueError("Density matrices must have shape (..., d, d)")
    low_precision = raw.dtype in (torch.float32, torch.complex64)
    tolerance = (2e-5 if low_precision else precision_for(raw.device).validation_atol) if atol is None else atol
    # MPS lacks complex Hermitian eigensolvers. Validation is detached by design,
    # so use CPU complex128 for reliable diagnostics without affecting gradients.
    state = raw.detach().cpu().to(torch.complex128)
    hermitian_error = (state - state.mH).abs().amax().item()
    trace_error = (state.diagonal(dim1=-2, dim2=-1).sum(-1).real - 1).abs().amax().item()
    eigenvalues = torch.linalg.eigvalsh((state + state.mH) / 2).real
    min_eigenvalue = eigenvalues.amin().item()
    purities = purity(state)
    dimension = state.shape[-1]
    min_purity = purities.amin().item()
    max_purity = purities.amax().item()
    valid = (
        hermitian_error <= tolerance
        and trace_error <= tolerance
        and min_eigenvalue >= -tolerance
        and min_purity >= 1 / dimension - tolerance
        and max_purity <= 1 + tolerance
    )
    report = {
        "valid": valid,
        "hermitian_error": hermitian_error,
        "trace_error": trace_error,
        "min_eigenvalue": min_eigenvalue,
        "min_purity": min_purity,
        "max_purity": max_purity,
        "atol": tolerance,
    }
    if not valid:
        message = f"Invalid density matrix: {report}"
        if raise_error:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return report
