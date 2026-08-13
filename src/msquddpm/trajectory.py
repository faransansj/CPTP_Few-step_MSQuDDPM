from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .states import validate_density_matrix


@dataclass
class Trajectory:
    states: dict[int, torch.Tensor]
    direction: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = {int(t): torch.as_tensor(rho) for t, rho in self.states.items()}

    def get_state(self, t: int) -> torch.Tensor:
        return self.states[int(t)]

    def validate(self) -> dict[int, dict]:
        return {t: validate_density_matrix(rho) for t, rho in self.states.items()}

    @property
    def steps(self) -> list[int]:
        return sorted(self.states)


def save_trajectory(trajectory: Trajectory, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": {int(t): rho.detach().cpu() for t, rho in trajectory.states.items()},
        "direction": trajectory.direction,
        "metadata": trajectory.metadata,
    }
    if path.suffix == ".pt":
        torch.save(payload, path)
    elif path.suffix == ".npz":
        arrays = {f"rho_{t}": rho.detach().cpu().numpy() for t, rho in trajectory.states.items()}
        arrays["steps"] = np.asarray(trajectory.steps)
        arrays["direction"] = np.asarray(trajectory.direction)
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError("Trajectory path must end in .pt or .npz")
    return path


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return Trajectory(payload["states"], payload["direction"], payload.get("metadata", {}))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            steps = [int(x) for x in data["steps"]]
            direction = str(data["direction"])
            states = {t: torch.from_numpy(data[f"rho_{t}"]) for t in steps}
        return Trajectory(states, direction)
    raise ValueError("Trajectory path must end in .pt or .npz")


def save_teacher_trajectory(
    forward: Trajectory,
    reverse: Trajectory,
    path: str | Path,
) -> Path:
    if forward.steps != reverse.steps:
        raise ValueError("Forward and reverse trajectories must contain the same steps")
    forward_shape = forward.get_state(0).shape
    reverse_shape = reverse.get_state(0).shape
    if forward_shape != reverse_shape or len(forward_shape) != 3:
        raise ValueError("Forward and reverse trajectories must have equal (batch, d, d) shapes")
    for t in forward.steps:
        if forward.get_state(t).shape != forward_shape or reverse.get_state(t).shape != reverse_shape:
            raise ValueError(f"Inconsistent trajectory shape at step {t}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = forward_shape[0]
    arrays: dict[str, np.ndarray] = {
        # Forward and reverse paths are independent ensemble samples, not pairs.
        "forward_sample_id": np.arange(count),
        "reverse_sample_id": np.arange(count),
        "sample_id": np.arange(count),  # Legacy positional index only.
        "paired": np.asarray(False),
    }
    for t in forward.steps:
        arrays[f"rho_{t}"] = forward.get_state(t).detach().cpu().numpy()
        arrays[f"reverse_rho_{t}"] = reverse.get_state(t).detach().cpu().numpy()
    arrays["steps"] = np.asarray(forward.steps)
    np.savez_compressed(path, **arrays)
    return path
