from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .datasets import make_dataset
from .forward_diffusion import ForwardDiffusion
from .metrics import nearest_metrics
from .precision import precision_for
from .reverse_model import ReverseMSQuDDPM
from .trainer import train_greedy
from .trajectory import save_teacher_trajectory, save_trajectory
from .utils import ensure_output_dirs, get_device, json_dump, set_seed


def train_experiment(config: dict) -> dict:
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
    teacher=output["trajectories"]/f"{name}_teacher.npz"; save_teacher_trajectory(forward,reverse,teacher)
    metrics=nearest_metrics(reverse.get_state(0).detach(),dataset)
    pd.DataFrame([{"experiment":name,"T":config["T"],**metrics}]).to_csv(output["metrics"]/f"{name}.csv",index=False)
    json_dump({"device":str(device),"real_dtype":str(model.theta.dtype),"complex_dtype":str(dataset.dtype),"checkpoint":str(checkpoint),"metrics":metrics},output["metrics"]/f"{name}_summary.json")
    return {"dataset":dataset,"forward":forward,"reverse":reverse,"model":model,"history":result.history,"metrics":metrics,"checkpoint":checkpoint,"outputs":output}


def load_experiment(checkpoint: str | Path, device: str = "auto") -> tuple[ReverseMSQuDDPM,dict,torch.Tensor]:
    # Always deserialize through CPU: checkpoints include float64 schedule metadata,
    # which cannot be materialized directly on MPS.
    target=get_device(device); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); config=payload["config"]
    model=ReverseMSQuDDPM(int(config["T"]),n_ancilla=int(config["n_ancilla"]),depth=int(config["depth"]),ancilla=config["ancilla"],seed=int(config["seed"]),init=config.get("init","normal"),device=target)
    model.load_state_dict(payload["model_state"])
    return model,config,payload["dataset"].to(device=target,dtype=precision_for(target).complex)
