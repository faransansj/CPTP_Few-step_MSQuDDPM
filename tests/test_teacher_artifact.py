import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from msquddpm.experiment import train_experiment
from msquddpm.provenance import sha256_file, validate_canonical_baseline_config
from msquddpm.trajectory import (
    TeacherArtifactError,
    Trajectory,
    load_teacher_trajectory,
    save_teacher_artifact,
)
from msquddpm.utils import load_config


def _trajectories(state: np.ndarray | None = None) -> tuple[Trajectory, Trajectory]:
    base = np.asarray(state if state is not None else [[[0.7, 0.0], [0.0, 0.3]], [[0.4, 0.0], [0.0, 0.6]]], dtype=np.complex128)
    forward = Trajectory({t: torch.from_numpy(base.copy()) for t in range(7)}, "forward")
    reverse = Trajectory({t: torch.from_numpy(base[::-1].copy()) for t in range(7)}, "reverse")
    return forward, reverse


def _save(path: Path, state: np.ndarray | None = None) -> Path:
    forward, reverse = _trajectories(state)
    return save_teacher_artifact(
        forward,
        reverse,
        path,
        dataset="clustered",
        seed=7,
        config_sha256="a" * 64,
        git_commit="b" * 40,
        checkpoint_sha256="c" * 64,
    )


def _rewrite(path: Path, change) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    change(arrays)
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def test_valid_teacher_v1_roundtrip_and_metadata(tmp_path):
    path = _save(tmp_path / "teacher.npz")
    teacher = load_teacher_trajectory(
        path,
        expected_dataset="clustered",
        expected_seed=7,
        expected_config_sha256="a" * 64,
        expected_checkpoint_sha256="c" * 64,
    )

    assert teacher.metadata == {
        "schema_version": 1,
        "dataset": "clustered",
        "seed": 7,
        "dtype": "complex128",
        "validation_tolerance": 1e-8,
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "checkpoint_sha256": "c" * 64,
    }
    assert teacher.forward.steps == teacher.reverse.steps == list(range(7))
    assert teacher.forward.get_state(3).shape == (2, 2, 2)
    assert teacher.physicality["passed"]
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)
        assert {f"rho_{t}" for t in range(7)} <= set(data.files)
        assert {f"reverse_rho_{t}" for t in range(7)} <= set(data.files)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.pop("rho_3"), "missing.*rho_3"),
        (lambda data: data.__setitem__("schema_version", np.asarray(2)), "schema_version"),
        (lambda data: data.__setitem__("rho_3", data["rho_3"][:1]), "rho_3.*shape"),
        (lambda data: data.__setitem__("rho_3", data["rho_3"].astype(np.complex64)), "rho_3.*complex128"),
        (lambda data: data["rho_3"].__setitem__((0, 0, 0), np.nan), "rho_3.*sample 0.*finite"),
        (lambda data: data["rho_3"].__setitem__((0, 0, 1), 0.2j), "rho_3.*sample 0.*Hermitian"),
        (lambda data: data["rho_3"].__setitem__((0, 0, 0), 0.8), "rho_3.*sample 0.*trace"),
        (lambda data: data["rho_3"].__setitem__((0, 0, 0), -0.1), "rho_3.*sample 0.*PSD"),
    ],
)
def test_teacher_loader_rejects_invalid_artifacts(tmp_path, change, message):
    path = _save(tmp_path / "teacher.npz")
    _rewrite(path, change)
    with pytest.raises(TeacherArtifactError, match=message):
        load_teacher_trajectory(path)


def test_teacher_loader_rejects_hash_mismatch_and_corruption(tmp_path):
    path = _save(tmp_path / "teacher.npz")
    with pytest.raises(TeacherArtifactError, match="config_sha256"):
        load_teacher_trajectory(path, expected_config_sha256="d" * 64)
    with pytest.raises(TeacherArtifactError, match="checkpoint_sha256"):
        load_teacher_trajectory(path, expected_checkpoint_sha256="e" * 64)

    path.write_bytes(b"not an npz")
    with pytest.raises(TeacherArtifactError, match="teacher.npz.*corrupt"):
        load_teacher_trajectory(path)


def test_teacher_tolerance_edge_passes_without_repair(tmp_path):
    state = np.asarray([[[-1e-10, 0.0], [0.0, 1.0 + 1e-10]]], dtype=np.complex128)
    path = _save(tmp_path / "teacher.npz", state)
    teacher = load_teacher_trajectory(path)
    loaded = teacher.forward.get_state(0).numpy()
    assert teacher.physicality["passed"]
    assert loaded[0, 0, 0] == -1e-10


def _tiny_provenance_config(tmp_path: Path, experiment: str, device: str = "cpu") -> Path:
    config = {
        "experiment": experiment,
        "dataset": "clustered",
        "seed": 5,
        "dataset_size": 2,
        "T": 6,
        "schedule": "cosine",
        "schedule_offset": 0.001,
        "n_ancilla": 1,
        "depth": 1,
        "ancilla": "zero",
        "loss": "mmd",
        "epochs": 1,
        "learning_rate": 0.01,
        "gamma": 1.0,
        "lr_decay_count": 1,
        "sampling_semantics": "official",
        "init": "normal",
        "device": device,
        "dtype": "complex128",
        "artifact_schema_version": 1,
        "provenance_manifest": True,
        "immutable_output": True,
        "output_root": str(tmp_path / experiment),
    }
    path = tmp_path / f"{experiment}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_smoke_serializer_loader_manifest_and_collision(tmp_path):
    config = load_config(_tiny_provenance_config(tmp_path, "synthetic"))
    result = train_experiment(config)
    root = Path(config["output_root"])
    manifest = json.loads((root / "manifest.json").read_text())

    assert manifest["manifest_schema_version"] == 1
    assert manifest["success"] is True and manifest["error"] is None
    assert manifest["git"]["commit"] and isinstance(manifest["git"]["dirty"], bool)
    assert manifest["config"]["sha256"] == sha256_file(root / "config.resolved.yaml")
    assert manifest["environment"]["python_version"]
    assert manifest["environment"]["uv_version"]
    assert manifest["environment"]["packages"]
    assert manifest["device"] == "cpu" and manifest["dtype"] == "complex128"
    assert manifest["dataset"] == "clustered" and manifest["seed"] == 5
    assert manifest["sampling_semantics"] == "official"
    assert manifest["runtime_seconds"] >= 0
    assert manifest["artifacts"]["checkpoint_sha256"] == sha256_file(result["checkpoint"])
    assert manifest["artifacts"]["trajectory_sha256"] == sha256_file(result["teacher"])
    assert manifest["physicality"]["passed"]
    assert "F_gen_0" in manifest["metrics"]
    load_teacher_trajectory(
        result["teacher"],
        expected_config_sha256=manifest["config"]["sha256"],
        expected_checkpoint_sha256=manifest["artifacts"]["checkpoint_sha256"],
    )

    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        train_experiment(config)
    assert (root / "manifest.json").read_bytes() == before


def test_failed_run_writes_manifest(tmp_path):
    config = load_config(_tiny_provenance_config(tmp_path, "failed", device="invalid"))
    with pytest.raises(Exception):
        train_experiment(config)
    manifest = json.loads((Path(config["output_root"]) / "manifest.json").read_text())
    assert manifest["success"] is False
    assert "invalid" in manifest["error"]


def test_canonical_cpu_configs_and_preflight():
    root = Path(__file__).parents[1]
    paths = sorted((root / "configs" / "baselines").glob("*.yaml"))
    assert [path.stem for path in paths] == [
        "circular_seed123",
        "circular_seed42",
        "circular_seed7",
        "clustered_seed123",
        "clustered_seed42",
        "clustered_seed7",
    ]
    output_roots = set()
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["canonical_baseline"] is True
        assert config["device"] == "cpu"
        assert config["dtype"] == "complex128"
        assert config["sampling_semantics"] == "official"
        assert config["artifact_schema_version"] == 1
        assert config["validation_tolerance"] == 1e-8
        assert config["seed"] in {7, 42, 123}
        assert config["output_root"] not in output_roots
        output_roots.add(config["output_root"])

    clustered = yaml.safe_load((root / "configs/baselines/clustered_seed7.yaml").read_text())
    validate_canonical_baseline_config(clustered, git_dirty=False)
    with pytest.raises(RuntimeError, match="clean Git"):
        validate_canonical_baseline_config(clustered, git_dirty=True)

    circular = yaml.safe_load((root / "configs/baselines/circular_seed7.yaml").read_text())
    assert circular["circular_acceptance_criterion"].startswith("TODO:")
    with pytest.raises(RuntimeError, match="Circular baseline evaluation metric"):
        validate_canonical_baseline_config(circular, git_dirty=False)
