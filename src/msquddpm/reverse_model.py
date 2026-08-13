from __future__ import annotations

import math

import torch
from torch import nn

from .precision import Precision, precision_for
from .states import validate_density_matrix
from .trajectory import Trajectory


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    # Explicit complex conversion avoids an MPS scalar-complex stacking bug.
    dtype = precision_for(theta.device).complex
    c = torch.complex(c, torch.zeros_like(c))
    minus_is = torch.complex(torch.zeros_like(s), -s).to(dtype)
    return torch.stack((torch.stack((c, minus_is)), torch.stack((minus_is, c))))


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(precision_for(theta.device).complex)


def _expand_gate(gate: torch.Tensor, qubit: int, total: int) -> torch.Tensor:
    result = torch.ones(1, 1, dtype=gate.dtype, device=gate.device)
    eye = torch.eye(2, dtype=gate.dtype, device=gate.device)
    for index in range(total):
        result = torch.kron(result, gate if index == qubit else eye)
    return result


def _cz(total: int, left: int, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    dtype = dtype or precision_for(device).complex
    values = torch.ones(2**total, dtype=dtype, device=device)
    for basis in range(2**total):
        bits = [(basis >> (total - 1 - q)) & 1 for q in range(total)]
        if bits[left] and bits[left + 1]:
            values[basis] = -1
    return torch.diag(values)


def _haar_ancilla(count: int, n_ancilla: int, generator: torch.Generator, device: torch.device, precision: Precision) -> torch.Tensor:
    real = torch.randn(count, 2, generator=generator, dtype=precision.real)
    imag = torch.randn(count, 2, generator=generator, dtype=precision.real)
    psi = torch.complex(real, imag)
    psi /= torch.linalg.vector_norm(psi, dim=1, keepdim=True)
    zeros = torch.zeros(count, 2**n_ancilla, dtype=precision.complex)
    basis_zero = torch.tensor([1, 0], dtype=precision.complex)
    for i in range(count):
        vector = psi[i]
        for _ in range(n_ancilla - 1):
            vector = torch.kron(vector, basis_zero)
        zeros[i] = vector
    return torch.einsum("bi,bj->bij", zeros, zeros.conj()).to(device)


def _zero_ancilla(count: int, n_ancilla: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    state = torch.zeros(count, 2**n_ancilla, 2**n_ancilla, dtype=dtype, device=device)
    state[:, 0, 0] = 1
    return state


class ReverseMSQuDDPM(nn.Module):
    """Original step-wise RX/RY/CZ PQCs with sampled ancilla measurements."""

    def __init__(
        self,
        steps: int,
        n_system: int = 1,
        n_ancilla: int = 2,
        depth: int = 4,
        ancilla: str = "haar",
        seed: int = 42,
        init: str = "normal",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        device = torch.device(device); precision = precision_for(device)
        self.steps, self.n_system, self.n_ancilla, self.depth = steps, n_system, n_ancilla, depth
        self.ancilla_type, self.seed = ancilla, seed
        total = n_system + n_ancilla
        generator = torch.Generator().manual_seed(seed)
        scale = 1.0 if init == "normal" else 1 / math.sqrt(total)
        parameters = torch.randn(steps, depth, total, 2, generator=generator, dtype=precision.real) * scale
        if init == "xavier":
            parameters[:, :, n_system:, :] = torch.randn(steps, depth, n_ancilla, 2, generator=generator, dtype=precision.real)
        self.theta = nn.Parameter(parameters.to(device))
        self._ancilla_generator = torch.Generator().manual_seed(seed + 1)
        self._measurement_generator = torch.Generator().manual_seed(seed + 2)

    @property
    def precision(self) -> Precision:
        return precision_for(self.theta.device)

    def _unitary(self, t: int) -> torch.Tensor:
        total = self.n_system + self.n_ancilla
        unitary = torch.eye(2**total, dtype=self.precision.complex, device=self.theta.device)
        for layer in range(self.depth):
            for qubit in range(total):
                unitary = _expand_gate(_rx(self.theta[t - 1, layer, qubit, 0]), qubit, total) @ unitary
                unitary = _expand_gate(_ry(self.theta[t - 1, layer, qubit, 1]), qubit, total) @ unitary
            for left in range(layer % 2, total - 1, 2):
                unitary = _cz(total, left, self.theta.device, self.precision.complex) @ unitary
        return unitary

    def _ancillas(self, count: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.ancilla_type == "zero":
            return _zero_ancilla(count, self.n_ancilla, device, self.precision.complex)
        if self.ancilla_type == "haar":
            return _haar_ancilla(count, self.n_ancilla, generator or self._ancilla_generator, device, self.precision)
        raise ValueError(f"Unknown ancilla type: {self.ancilla_type}")

    def reverse_step(self, rho: torch.Tensor, t: int, validate: bool = True, ancilla_states: torch.Tensor | None = None, measurement_generator: torch.Generator | None = None) -> torch.Tensor:
        if not 1 <= t <= self.steps:
            raise ValueError(f"t must be in [1, {self.steps}]")
        if rho.device != self.theta.device:
            raise ValueError(f"rho is on {rho.device}, model is on {self.theta.device}")
        batch = len(rho)
        rng = measurement_generator or self._measurement_generator
        ancilla = ancilla_states if ancilla_states is not None else self._ancillas(batch, rho.device)
        if ancilla.shape != (batch, 2**self.n_ancilla, 2**self.n_ancilla):
            raise ValueError("ancilla_states has incompatible batch or matrix shape")
        joint = torch.einsum("aij,akl->aikjl", rho.to(self.precision.complex), ancilla).reshape(batch, 2 ** (self.n_system + self.n_ancilla), 2 ** (self.n_system + self.n_ancilla))
        unitary = self._unitary(t)
        evolved = unitary[None] @ joint @ unitary.mH[None]
        ds, da = 2**self.n_system, 2**self.n_ancilla
        blocks = evolved.reshape(batch, ds, da, ds, da)
        probabilities = torch.stack([blocks[:, :, outcome, :, outcome].diagonal(dim1=-2, dim2=-1).sum(-1).real for outcome in range(da)], dim=1).clamp_min(0)
        probabilities /= probabilities.sum(1, keepdim=True).clamp_min(1e-7 if self.precision.real == torch.float32 else 1e-15)
        probabilities = torch.nan_to_num(probabilities.detach(), nan=1 / da, posinf=1 / da, neginf=0.0)
        probabilities /= probabilities.sum(1, keepdim=True).clamp_min(1e-7 if self.precision.real == torch.float32 else 1e-15)
        # CPU categorical sampling is intentional and deterministic; only tiny
        # probability/outcome tensors cross the boundary, core circuit stays MPS.
        outcomes = torch.multinomial(probabilities.cpu(), 1, generator=rng).squeeze(1).to(rho.device)
        result = torch.stack([blocks[i, :, outcomes[i], :, outcomes[i]] for i in range(batch)])
        normalizer = result.diagonal(dim1=-2, dim2=-1).sum(-1).real.clamp_min(1e-7 if self.precision.real == torch.float32 else 1e-15)
        result /= normalizer[:, None, None]
        if validate:
            validate_density_matrix(result.detach())
        return result

    def generate(self, rho_t: torch.Tensor, return_trajectory: bool = False):
        states = {self.steps: rho_t}; current = rho_t
        for t in range(self.steps, 0, -1):
            current = self.reverse_step(current, t); states[t - 1] = current
        trajectory = Trajectory(states, "reverse", {"ancilla": self.ancilla_type})
        self.last_trajectory = trajectory
        return trajectory if return_trajectory else current

    def get_state(self, t: int, trajectory: Trajectory | None = None) -> torch.Tensor:
        source = trajectory or getattr(self, "last_trajectory", None)
        if source is None:
            raise RuntimeError("No generated trajectory; call generate(return_trajectory=True) first")
        return source.get_state(t)
