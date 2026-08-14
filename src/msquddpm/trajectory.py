from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .states import validate_density_matrix

TEACHER_SCHEMA_VERSION = 1
TEACHER_STEPS = list(range(7))  # T=6 teacher chains: steps 0..6.
TEACHER_TOLERANCE = 1e-8


def _require_hex(value: str, lengths: set[int], label: str, path: Path) -> None:
    if not isinstance(value, str) or len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        expected = "/".join(str(length) for length in sorted(lengths))
        raise TeacherArtifactError(f"{path.name}: {label} must be a {expected}-character hexadecimal digest")


class TeacherArtifactError(Exception):
    """Strict Teacher schema v1 violation. Artifacts are never repaired."""


@dataclass
class TeacherTrajectory:
    metadata: dict
    forward: Trajectory
    reverse: Trajectory
    physicality: dict


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


def _check_teacher_physicality(
    arrays: dict[str, np.ndarray], steps: list[int], tolerance: float, path: Path
) -> dict:
    """Per-sample physicality gate: finite, Hermitian, PSD, unit trace. No repair."""
    report = {
        "passed": True,
        "tolerance": float(tolerance),
        "max_hermitian_error": 0.0,
        "max_trace_error": 0.0,
        "min_eigenvalue": float("inf"),
        "num_failed_states": 0,
        "failed_indices": [],
        "steps": {},
    }
    for key in [f"rho_{t}" for t in steps] + [f"reverse_rho_{t}" for t in steps]:
        step = int(key.rsplit("_", 1)[1])
        rho = arrays[key]
        step_report = {
            "passed": True,
            "max_hermitian_error": 0.0,
            "max_trace_error": 0.0,
            "min_eigenvalue": float("inf"),
            "num_failed_states": 0,
            "failed_indices": [],
        }
        report["steps"][key] = step_report
        for i in range(rho.shape[0]):
            sample = rho[i]
            label = f"{path.name}: key {key} step {step} sample {i}"
            if not np.isfinite(sample).all():
                raise TeacherArtifactError(f"{label}: not finite")
            hermitian_error = float(np.abs(sample - sample.conj().T).max())
            step_report["max_hermitian_error"] = max(step_report["max_hermitian_error"], hermitian_error)
            report["max_hermitian_error"] = max(report["max_hermitian_error"], hermitian_error)
            if hermitian_error > tolerance:
                raise TeacherArtifactError(f"{label}: not Hermitian (error {hermitian_error:.3e} > {tolerance})")
            min_eigenvalue = float(np.linalg.eigvalsh((sample + sample.conj().T) / 2).real.min())
            step_report["min_eigenvalue"] = min(step_report["min_eigenvalue"], min_eigenvalue)
            report["min_eigenvalue"] = min(report["min_eigenvalue"], min_eigenvalue)
            if min_eigenvalue < -tolerance:
                raise TeacherArtifactError(f"{label}: not PSD (min eigenvalue {min_eigenvalue:.3e} < {-tolerance})")
            trace_error = abs(complex(np.trace(sample)) - 1.0)
            step_report["max_trace_error"] = max(step_report["max_trace_error"], trace_error)
            report["max_trace_error"] = max(report["max_trace_error"], trace_error)
            if trace_error > tolerance:
                raise TeacherArtifactError(f"{label}: trace error {trace_error:.3e} > {tolerance}")
    return report


def save_teacher_artifact(
    forward: Trajectory,
    reverse: Trajectory,
    path: str | Path,
    *,
    dataset: str,
    seed: int,
    config_sha256: str,
    git_commit: str,
    checkpoint_sha256: str,
    tolerance: float = TEACHER_TOLERANCE,
) -> Path:
    """Strict Teacher schema v1 serializer. Validates before writing; never repairs."""
    path = Path(path)
    if type(seed) is not int:
        raise TeacherArtifactError(f"{path.name}: seed must be an integer scalar")
    if not isinstance(dataset, str):
        raise TeacherArtifactError(f"{path.name}: dataset must be a string scalar")
    if tolerance != TEACHER_TOLERANCE:
        raise TeacherArtifactError(
            f"{path.name}: validation_tolerance {tolerance} invalid (expected {TEACHER_TOLERANCE})"
        )
    if dataset not in {"clustered", "circular"}:
        raise TeacherArtifactError(f"{path.name}: unsupported dataset {dataset!r}")
    _require_hex(config_sha256, {64}, "config_sha256", path)
    _require_hex(checkpoint_sha256, {64}, "checkpoint_sha256", path)
    _require_hex(git_commit, {40, 64}, "git_commit", path)
    if forward.steps != TEACHER_STEPS or reverse.steps != TEACHER_STEPS:
        raise TeacherArtifactError(
            f"{path.name}: teacher schema v1 requires steps {TEACHER_STEPS} for both chains; "
            f"got forward {forward.steps}, reverse {reverse.steps}"
        )
    arrays: dict[str, np.ndarray] = {}
    batch = None
    for prefix, chain in (("rho", forward), ("reverse_rho", reverse)):
        for t in TEACHER_STEPS:
            key = f"{prefix}_{t}"
            state = chain.get_state(t).detach().cpu()
            if state.dtype != torch.complex128:
                raise TeacherArtifactError(f"{path.name}: key {key}: dtype {state.dtype} is not complex128")
            try:
                array = state.resolve_conj().resolve_neg().contiguous().numpy()
            except (RuntimeError, TypeError) as exc:
                raise TeacherArtifactError(
                    f"{path.name}: key {key}: cannot materialize tensor as a complex128 NumPy array ({exc})"
                ) from exc
            if array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (2, 2):
                raise TeacherArtifactError(
                    f"{path.name}: key {key}: shape {array.shape} invalid "
                    "(expected non-empty (batch, 2, 2))"
                )
            if batch is None:
                batch = array.shape[0]
            elif array.shape[0] != batch:
                raise TeacherArtifactError(
                    f"{path.name}: key {key}: shape {array.shape} invalid (batch mismatch, expected {batch})"
                )
            arrays[key] = array
    _check_teacher_physicality(arrays, TEACHER_STEPS, tolerance, path)
    arrays.update(
        {
            "schema_version": np.asarray(TEACHER_SCHEMA_VERSION),
            "dataset": np.asarray(str(dataset)),
            "seed": np.asarray(int(seed)),
            "dtype": np.asarray("complex128"),
            "validation_tolerance": np.asarray(float(tolerance)),
            "config_sha256": np.asarray(str(config_sha256)),
            "git_commit": np.asarray(str(git_commit)),
            "checkpoint_sha256": np.asarray(str(checkpoint_sha256)),
            "steps": np.asarray(TEACHER_STEPS),
            # Forward and reverse paths are independent ensemble samples, not pairs.
            "forward_sample_id": np.arange(batch),
            "reverse_sample_id": np.arange(batch),
            "sample_id": np.arange(batch),  # Legacy positional index only.
            "paired": np.asarray(False),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with temporary:
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise
    return path


def load_teacher_trajectory(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
    expected_config_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> TeacherTrajectory:
    """Strict Teacher schema v1 loader. Raises TeacherArtifactError on any violation."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception as exc:
        raise TeacherArtifactError(f"{path.name}: corrupt or unreadable NPZ ({exc})") from exc

    def scalar(key: str, kind: str):
        if key not in arrays:
            raise TeacherArtifactError(f"{path.name}: missing required key {key}")
        array = arrays[key]
        valid_kind = {
            "integer": np.issubdtype(array.dtype, np.integer) and not np.issubdtype(array.dtype, np.bool_),
            "float": np.issubdtype(array.dtype, np.floating),
            "string": array.dtype.kind == "U",
            "boolean": np.issubdtype(array.dtype, np.bool_),
        }[kind]
        if array.ndim != 0 or not valid_kind:
            raise TeacherArtifactError(
                f"{path.name}: key {key} must be a scalar {kind} array without coercion, "
                f"got {array.dtype}{array.shape}"
            )
        return array.item()

    schema_version = scalar("schema_version", "integer")
    if schema_version != TEACHER_SCHEMA_VERSION:
        raise TeacherArtifactError(
            f"{path.name}: schema_version {schema_version} unsupported (expected {TEACHER_SCHEMA_VERSION})"
        )
    metadata = {
        "schema_version": schema_version,
        "dataset": scalar("dataset", "string"),
        "seed": scalar("seed", "integer"),
        "dtype": scalar("dtype", "string"),
        "validation_tolerance": scalar("validation_tolerance", "float"),
        "config_sha256": scalar("config_sha256", "string"),
        "git_commit": scalar("git_commit", "string"),
        "checkpoint_sha256": scalar("checkpoint_sha256", "string"),
    }
    if metadata["dataset"] not in {"clustered", "circular"}:
        raise TeacherArtifactError(f"{path.name}: unsupported dataset {metadata['dataset']!r}")
    _require_hex(metadata["config_sha256"], {64}, "config_sha256", path)
    _require_hex(metadata["checkpoint_sha256"], {64}, "checkpoint_sha256", path)
    _require_hex(metadata["git_commit"], {40, 64}, "git_commit", path)
    if metadata["dtype"] != "complex128":
        raise TeacherArtifactError(f"{path.name}: metadata dtype {metadata['dtype']!r} is not complex128")
    if metadata["validation_tolerance"] != TEACHER_TOLERANCE:
        raise TeacherArtifactError(
            f"{path.name}: validation_tolerance {metadata['validation_tolerance']} invalid "
            f"(expected {TEACHER_TOLERANCE})"
        )
    paired = scalar("paired", "boolean")
    if paired:
        raise TeacherArtifactError(f"{path.name}: paired must be false for independent forward/reverse ensembles")
    if "steps" not in arrays:
        raise TeacherArtifactError(f"{path.name}: missing required key steps")
    step_array = arrays["steps"]
    if (
        step_array.shape != (len(TEACHER_STEPS),)
        or not np.issubdtype(step_array.dtype, np.integer)
        or np.issubdtype(step_array.dtype, np.bool_)
        or not np.array_equal(step_array, np.asarray(TEACHER_STEPS))
    ):
        raise TeacherArtifactError(
            f"{path.name}: steps dtype/shape/values invalid "
            f"(got {step_array.dtype}{step_array.shape}, expected integer {TEACHER_STEPS})"
        )
    steps = TEACHER_STEPS
    for name, expected in (
        ("dataset", expected_dataset),
        ("seed", expected_seed),
        ("config_sha256", expected_config_sha256),
        ("git_commit", expected_git_commit),
        ("checkpoint_sha256", expected_checkpoint_sha256),
    ):
        if expected is not None and metadata[name] != expected:
            raise TeacherArtifactError(
                f"{path.name}: {name} mismatch: expected {expected}, found {metadata[name]}"
            )

    batch = None
    for prefix in ("rho", "reverse_rho"):
        for t in TEACHER_STEPS:
            key = f"{prefix}_{t}"
            if key not in arrays:
                raise TeacherArtifactError(f"{path.name}: missing required key {key}")
            array = arrays[key]
            if array.dtype == object:
                raise TeacherArtifactError(f"{path.name}: key {key}: pickled object arrays are forbidden")
            if array.dtype != np.complex128:
                raise TeacherArtifactError(f"{path.name}: key {key}: dtype {array.dtype} is not complex128")
            if array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (2, 2):
                raise TeacherArtifactError(
                    f"{path.name}: key {key}: shape {array.shape} invalid "
                    "(expected non-empty (batch, 2, 2))"
                )
            if batch is None:
                batch = array.shape[0]
            elif array.shape[0] != batch:
                raise TeacherArtifactError(
                    f"{path.name}: key {key}: shape {array.shape} invalid (batch mismatch, expected {batch})"
                )
    expected_ids = np.arange(batch)
    for key in ("forward_sample_id", "reverse_sample_id", "sample_id"):
        ids = arrays.get(key)
        if (
            ids is None
            or ids.shape != (batch,)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.issubdtype(ids.dtype, np.bool_)
            or not np.array_equal(ids, expected_ids)
        ):
            dtype_shape = "missing" if ids is None else f"{ids.dtype}{ids.shape}"
            raise TeacherArtifactError(
                f"{path.name}: key {key} must be integer contiguous sample ids 0..{batch - 1}; "
                f"got {dtype_shape}"
            )
    physicality = _check_teacher_physicality(arrays, TEACHER_STEPS, metadata["validation_tolerance"], path)
    forward = Trajectory({t: torch.from_numpy(arrays[f"rho_{t}"]) for t in TEACHER_STEPS}, "forward", metadata)
    reverse = Trajectory(
        {t: torch.from_numpy(arrays[f"reverse_rho_{t}"]) for t in TEACHER_STEPS}, "reverse", metadata
    )
    return TeacherTrajectory(metadata=metadata, forward=forward, reverse=reverse, physicality=physicality)
