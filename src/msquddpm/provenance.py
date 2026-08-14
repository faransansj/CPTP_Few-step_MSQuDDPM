"""Centralized provenance helpers: hashes, Git identity, environment, canonical preflight."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json_atomic(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with temporary:
            json.dump(data, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def git_provenance() -> dict:
    """Commit, dirty flag, and a concise dirty identity (hash of status + diff)."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain")
        dirty = bool(status.strip())
        dirty_identity = (
            hashlib.sha256((status + _git("diff", "HEAD")).encode()).hexdigest()[:16] if dirty else None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Unknown is treated as dirty so canonical runs stay blocked.
        commit, dirty, dirty_identity = "", True, "unavailable"
    return {"commit": commit, "dirty": dirty, "dirty_identity": dirty_identity}


def peak_rss_bytes() -> int | None:
    """Process peak RSS normalized to bytes where the stdlib exposes it."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def environment() -> dict:
    try:
        uv_version = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        uv_version = ""
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib_metadata.distributions()
        if dist.metadata["Name"]
    }
    lock = REPO_ROOT / "uv.lock"
    return {
        "python_version": platform.python_version(),
        "uv_version": uv_version,
        "packages": packages,
        "platform": platform.platform(),
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
    }


def validate_canonical_baseline_config(config: dict, git_dirty: bool) -> None:
    """Preflight for canonical baseline runs. No-op for non-canonical configs."""
    if not config.get("canonical_baseline"):
        return
    if git_dirty:
        raise RuntimeError(
            "Canonical baseline runs require a clean Git worktree; commit or stash changes first"
        )
    requirements = {
        "device": "cpu",
        "dtype": "complex128",
        "sampling_semantics": "official",
        "artifact_schema_version": 1,
        "validation_tolerance": 1e-8,
        "provenance_manifest": True,
        "immutable_output": True,
        "T": 6,
    }
    for key, expected in requirements.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Canonical baseline config requires {key}={expected!r}, got {config.get(key)!r}"
            )
    dataset = config.get("dataset")
    if dataset not in {"clustered", "circular"} or config.get("seed") not in {7, 42, 123}:
        raise RuntimeError("Canonical baseline requires clustered/circular with seed 7, 42, or 123")
    if dataset == "circular":
        criterion = str(config.get("circular_acceptance_criterion", ""))
        if not criterion or criterion.startswith("TODO:"):
            raise RuntimeError(
                "Circular baseline evaluation metric is undefined: circular_acceptance_criterion "
                "still contains a TODO; canonical circular runs are blocked until it is resolved"
            )
    if dataset == "clustered":
        criterion = config.get("cluster_acceptance", {})
        expected = {
            "metric": "F_gen_0",
            "aggregation": "mean",
            "seeds": [7, 42, 123],
            "minimum": 0.95,
            "per_seed_pass_fail": False,
        }
        if criterion != expected:
            raise RuntimeError(
                "Clustered canonical acceptance must be the structured three-seed aggregate "
                "mean F_gen_0 >= 0.95 policy, never a per-seed pass/fail"
            )
