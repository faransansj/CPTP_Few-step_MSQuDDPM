import json
from pathlib import Path
import subprocess
import sys

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
        expected_git_commit="b" * 40,
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
    assert teacher.physicality["num_failed_states"] == 0
    assert teacher.physicality["failed_indices"] == []
    assert set(teacher.physicality["steps"]) == {f"rho_{t}" for t in range(7)} | {
        f"reverse_rho_{t}" for t in range(7)
    }
    with np.load(path, allow_pickle=False) as data:
        assert all(data[key].dtype != object for key in data.files)
        assert {f"rho_{t}" for t in range(7)} <= set(data.files)
        assert {f"reverse_rho_{t}" for t in range(7)} <= set(data.files)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.pop("rho_3"), "missing.*rho_3"),
        (lambda data: data.__setitem__("schema_version", np.asarray(2)), "schema_version"),
        (lambda data: data.__setitem__("schema_version", np.asarray(1.9)), "schema_version.*integer"),
        (lambda data: data.__setitem__("seed", np.asarray(True)), "seed.*integer"),
        (lambda data: data.__setitem__("steps", np.arange(7, dtype=float) + 0.1), "steps.*invalid"),
        (lambda data: data.__setitem__("sample_id", np.arange(2, dtype=float)), "sample_id.*integer"),
        (lambda data: data.__setitem__("paired", np.asarray(0)), "paired.*boolean"),
        (lambda data: data.__setitem__("validation_tolerance", np.asarray(1e-4)), "validation_tolerance"),
        (lambda data: data.__setitem__("paired", np.asarray(True)), "paired"),
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


def test_teacher_saver_materializes_valid_conjugate_views(tmp_path):
    forward, reverse = _trajectories()
    forward = Trajectory({t: state.conj() for t, state in forward.states.items()}, "forward")
    path = save_teacher_artifact(
        forward,
        reverse,
        tmp_path / "teacher.npz",
        dataset="clustered",
        seed=7,
        config_sha256="a" * 64,
        git_commit="b" * 40,
        checkpoint_sha256="c" * 64,
    )
    loaded = load_teacher_trajectory(path)
    assert torch.equal(loaded.forward.get_state(0), forward.get_state(0).resolve_conj())


def test_teacher_saver_rejects_coercion_without_overwriting(tmp_path):
    path = tmp_path / "teacher.npz"
    path.write_bytes(b"keep")
    forward, reverse = _trajectories()
    with pytest.raises(TeacherArtifactError, match="seed must be an integer"):
        save_teacher_artifact(
            forward,
            reverse,
            path,
            dataset="clustered",
            seed=7.0,
            config_sha256="a" * 64,
            git_commit="b" * 40,
            checkpoint_sha256="c" * 64,
        )
    assert path.read_bytes() == b"keep"


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
    assert manifest["config"]["path"] == "config.resolved.yaml"
    assert manifest["config"]["sha256"] == sha256_file(root / "config.resolved.yaml")
    assert manifest["config"]["source_path"] == config["config_path"]
    assert manifest["config"]["source_sha256"] == sha256_file(config["config_path"])
    assert "config_path" not in yaml.safe_load((root / "config.resolved.yaml").read_text())
    assert manifest["environment"]["python_version"]
    assert manifest["environment"]["uv_version"]
    assert manifest["environment"]["packages"]
    assert manifest["device"] == "cpu" and manifest["dtype"] == "complex128"
    assert manifest["dataset"] == "clustered" and manifest["seed"] == 5
    assert manifest["sampling_semantics"] == "official"
    assert manifest["start_time"].endswith("+00:00")
    assert manifest["end_time"].endswith("+00:00")
    assert manifest["runtime_seconds"] >= 0
    assert manifest["peak_rss_bytes"] is None or manifest["peak_rss_bytes"] > 0
    assert manifest["parent_run_id"] is None
    assert manifest["artifacts"]["checkpoint_sha256"] == sha256_file(result["checkpoint"])
    assert manifest["artifacts"]["trajectory_sha256"] == sha256_file(result["teacher"])
    assert set(manifest["artifacts"]["files"]) == {
        "checkpoint", "teacher", "forward_pt", "forward_npz", "reverse_pt", "reverse_npz",
        "history", "metrics", "summary",
    }
    for artifact in manifest["artifacts"]["files"].values():
        assert artifact["sha256"] == sha256_file(artifact["path"])
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


def test_immutable_output_rejects_existing_directory(tmp_path):
    config = load_config(_tiny_provenance_config(tmp_path, "occupied"))
    root = Path(config["output_root"])
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        train_experiment(config)
    assert sentinel.read_text() == "do not overwrite"


def test_failed_run_writes_manifest(tmp_path):
    config = load_config(_tiny_provenance_config(tmp_path, "failed", device="invalid"))
    with pytest.raises(Exception):
        train_experiment(config)
    manifest = json.loads((Path(config["output_root"]) / "manifest.json").read_text())
    assert manifest["success"] is False
    assert "invalid" in manifest["error"]
    assert manifest["metrics"] is None
    assert manifest["physicality"] is None
    assert manifest["artifacts"] == {}
    assert manifest["start_time"].endswith("+00:00")
    assert manifest["end_time"].endswith("+00:00")


def test_base_exception_writes_failure_manifest(tmp_path, monkeypatch):
    config = load_config(_tiny_provenance_config(tmp_path, "interrupted"))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("stopped")

    monkeypatch.setattr("msquddpm.experiment._run_training", interrupt)
    with pytest.raises(KeyboardInterrupt, match="stopped"):
        train_experiment(config)
    manifest = json.loads((Path(config["output_root"]) / "manifest.json").read_text())
    assert manifest["success"] is False
    assert manifest["error"] == "KeyboardInterrupt: stopped"
    assert manifest["end_time"].endswith("+00:00")


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
        assert config["parent_run_id"] is None
        assert config["seed"] in {7, 42, 123}
        assert config["output_root"] not in output_roots
        output_roots.add(config["output_root"])
        validate_canonical_baseline_config(config, git_dirty=False)

    clustered = yaml.safe_load((root / "configs/baselines/clustered_seed7.yaml").read_text())
    assert clustered["cluster_acceptance"] == {
        "metric": "F_gen_0",
        "aggregation": "mean",
        "seeds": [7, 42, 123],
        "minimum": 0.95,
        "per_seed_pass_fail": False,
    }
    validate_canonical_baseline_config(clustered, git_dirty=False)
    with pytest.raises(RuntimeError, match="clean Git"):
        validate_canonical_baseline_config(clustered, git_dirty=True)

    circular = yaml.safe_load((root / "configs/baselines/circular_seed7.yaml").read_text())
    assert circular["circular_acceptance"] == {
        "metric": "wasserstein",
        "aggregation": "mean",
        "seeds": [7, 42, 123],
        "maximum": 0.020,
        "per_seed_pass_fail": False,
    }
    validate_canonical_baseline_config(circular, git_dirty=False)
    invalid_circular = {
        **circular,
        "circular_acceptance": {**circular["circular_acceptance"], "maximum": 0.021},
    }
    with pytest.raises(RuntimeError, match="mean Wasserstein <= 0.020"):
        validate_canonical_baseline_config(invalid_circular, git_dirty=False)


def test_sha256_known_value_and_validation_cli_missing_input(tmp_path):
    file = tmp_path / "value.txt"
    file.write_bytes(b"abc")
    assert sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    script = Path(__file__).parents[1] / "scripts" / "validate_teacher.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.npz")], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "INVALID" in result.stderr and "missing.npz" in result.stderr
