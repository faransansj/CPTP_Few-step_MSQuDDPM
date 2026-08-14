from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from .datasets import make_dataset
from .forward_diffusion import ForwardDiffusion
from .metrics import nearest_metrics
from .precision import precision_for
from .provenance import (
    environment,
    git_provenance,
    peak_rss_bytes,
    sha256_file,
    validate_canonical_baseline_config,
    write_json_atomic,
)
from .reverse_model import ReverseMSQuDDPM
from .trainer import train_greedy
from .trajectory import (
    load_teacher_trajectory,
    save_teacher_artifact,
    save_teacher_trajectory,
    save_trajectory,
)
from .utils import ensure_output_dirs, get_device, json_dump, set_seed


def train_experiment(config: dict) -> dict:
    if config.get("artifact_schema_version") == 1 or config.get("provenance_manifest"):
        return _train_experiment_provenanced(config)
    return _run_training(config)


def _run_training(config: dict, teacher_context: dict | None = None) -> dict:
    seed = int(config["seed"]); set_seed(seed)
    device = get_device(config.get("device", "auto")); output = ensure_output_dirs(config.get("output_root", "outputs"))
    name = config["experiment"]
    dataset = make_dataset(config["dataset"], int(config["dataset_size"]), seed, device)
    diffusion = ForwardDiffusion(int(config["T"]), config["schedule"], float(config.get("schedule_offset", 0.001)))
    forward = diffusion.diffuse(dataset)
    model = ReverseMSQuDDPM(
        steps=int(config["T"]), n_ancilla=int(config["n_ancilla"]), depth=int(config["depth"]),
        ancilla=config["ancilla"], seed=seed, init=config.get("init", "normal"), device=device
    )
    result = train_greedy(
        model, forward, int(config["epochs"]), float(config["learning_rate"]), config["loss"],
        float(config.get("gamma", 1.0)), sampling_semantics=config.get("sampling_semantics", "official"),
        lr_decay_count=int(config.get("lr_decay_count", 2))
    )
    mixed = torch.eye(2,dtype=precision_for(device).complex,device=device)[None].repeat(len(dataset),1,1)/2
    reverse = model.generate(mixed, return_trajectory=True)
    checkpoint = output["checkpoints"] / f"{name}.pt"
    torch.save({"model_state":model.state_dict(),"config":config,"betas":diffusion.betas,"dataset":dataset.detach().cpu()},checkpoint)
    history_path=output["histories"]/f"{name}.csv"; result.history.to_csv(history_path,index=False)
    forward_pt=output["trajectories"]/f"{name}_forward.pt"; reverse_pt=output["trajectories"]/f"{name}_reverse.pt"
    save_trajectory(forward,forward_pt); save_trajectory(reverse,reverse_pt)
    save_trajectory(forward,forward_pt.with_suffix('.npz')); save_trajectory(reverse,reverse_pt.with_suffix('.npz'))
    teacher=output["trajectories"]/f"{name}_teacher.npz"
    if teacher_context is None:
        save_teacher_trajectory(forward,reverse,teacher)
    else:
        save_teacher_artifact(forward,reverse,teacher,checkpoint_sha256=sha256_file(checkpoint),**teacher_context)
    metrics=nearest_metrics(reverse.get_state(0).detach(),dataset)
    pd.DataFrame([{"experiment":name,"T":config["T"],**metrics}]).to_csv(output["metrics"]/f"{name}.csv",index=False)
    json_dump({"device":str(device),"real_dtype":str(model.theta.dtype),"complex_dtype":str(dataset.dtype),"checkpoint":str(checkpoint),"metrics":metrics},output["metrics"]/f"{name}_summary.json")
    return {"dataset":dataset,"forward":forward,"reverse":reverse,"model":model,"history":result.history,"metrics":metrics,"checkpoint":checkpoint,"teacher":teacher,"outputs":output}


def _train_experiment_provenanced(config: dict) -> dict:
    """Opt-in wrapper: exclusively claim output, then always leave an atomic manifest."""
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    root = Path(config.get("output_root", "outputs"))
    immutable = bool(config.get("immutable_output", True))
    try:
        root.mkdir(parents=True, exist_ok=not immutable)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Immutable output directory already exists: {root}; refusing to overwrite"
        ) from exc
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable output manifest already exists: {manifest_path}")

    manifest: dict = {
        "manifest_schema_version": 1,
        "experiment": config.get("experiment"),
        "dataset": config.get("dataset"),
        "seed": config.get("seed"),
        "sampling_semantics": config.get("sampling_semantics", "official"),
        "parent_run_id": config.get("parent_run_id"),
        "device": str(config.get("device", "auto")),
        "dtype": str(config.get("dtype", "")),
        "start_time": started_at.isoformat(),
        "git": None,
        "environment": None,
        "config": None,
        "metrics": None,
        "physicality": None,
        "artifacts": {},
        "success": False,
        "error": "Run did not complete",
    }
    result = None
    try:
        git = git_provenance()
        manifest["git"] = git
        resolved_config = {key: value for key, value in config.items() if key != "config_path"}
        config_yaml = root / "config.resolved.yaml"
        config_yaml.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
        source_config = Path(str(config.get("config_path", "")))
        manifest["config"] = {
            "path": config_yaml.name,
            "sha256": sha256_file(config_yaml),
            "source_path": str(source_config) if source_config.is_file() else None,
            "source_sha256": sha256_file(source_config) if source_config.is_file() else None,
        }
        manifest["environment"] = environment()
        validate_canonical_baseline_config(config, git_dirty=git["dirty"])
        if type(config.get("artifact_schema_version", 1)) is not int or config.get(
            "artifact_schema_version", 1
        ) != 1:
            raise ValueError("artifact_schema_version must be integer 1")
        if float(config.get("validation_tolerance", 1e-8)) != 1e-8:
            raise ValueError("validation_tolerance must be 1e-8 for Teacher schema v1")

        result = _run_training(
            config,
            teacher_context={
                "dataset": str(config["dataset"]),
                "seed": int(config["seed"]),
                "config_sha256": manifest["config"]["sha256"],
                "git_commit": git["commit"],
                "tolerance": float(config.get("validation_tolerance", 1e-8)),
            },
        )
        checkpoint_sha256 = sha256_file(result["checkpoint"])
        loaded = load_teacher_trajectory(
            result["teacher"],
            expected_dataset=str(config["dataset"]),
            expected_seed=int(config["seed"]),
            expected_config_sha256=manifest["config"]["sha256"],
            expected_git_commit=git["commit"],
            expected_checkpoint_sha256=checkpoint_sha256,
        )
        name, output = config["experiment"], result["outputs"]
        files = {
            "checkpoint": result["checkpoint"],
            "teacher": result["teacher"],
            "forward_pt": output["trajectories"] / f"{name}_forward.pt",
            "forward_npz": output["trajectories"] / f"{name}_forward.npz",
            "reverse_pt": output["trajectories"] / f"{name}_reverse.pt",
            "reverse_npz": output["trajectories"] / f"{name}_reverse.npz",
            "history": output["histories"] / f"{name}.csv",
            "metrics": output["metrics"] / f"{name}.csv",
            "summary": output["metrics"] / f"{name}_summary.json",
        }
        manifest.update(
            {
                "success": True,
                "error": None,
                "device": str(result["dataset"].device),
                "dtype": str(result["dataset"].dtype).replace("torch.", ""),
                "metrics": result["metrics"],
                "physicality": loaded.physicality,
                "artifacts": {
                    "checkpoint_sha256": checkpoint_sha256,
                    "trajectory_sha256": sha256_file(result["teacher"]),
                    "files": {
                        label: {"path": str(path), "sha256": sha256_file(path)}
                        for label, path in files.items()
                    },
                },
            }
        )
    except BaseException as exc:
        manifest.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        manifest["end_time"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["peak_rss_bytes"] = peak_rss_bytes()
        write_json_atomic(manifest, manifest_path)
    result["manifest"] = manifest_path
    return result


def load_experiment(checkpoint: str | Path, device: str = "auto") -> tuple[ReverseMSQuDDPM,dict,torch.Tensor]:
    # Always deserialize through CPU: checkpoints include float64 schedule metadata,
    # which cannot be materialized directly on MPS.
    target=get_device(device); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); config=payload["config"]
    model=ReverseMSQuDDPM(int(config["T"]),n_ancilla=int(config["n_ancilla"]),depth=int(config["depth"]),ancilla=config["ancilla"],seed=int(config["seed"]),init=config.get("init","normal"),device=target)
    model.load_state_dict(payload["model_state"])
    return model,config,payload["dataset"].to(device=target,dtype=precision_for(target).complex)
