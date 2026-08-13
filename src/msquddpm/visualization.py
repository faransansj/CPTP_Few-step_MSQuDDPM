from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics import nearest_metrics
from .states import density_to_bloch, purity


def _sphere(ax) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 16j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.8", linewidth=0.3)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_box_aspect((1, 1, 1))


def _scatter(ax, states, label=None, marker="o", alpha=0.7) -> None:
    xyz = density_to_bloch(states).detach().cpu().numpy()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=13, marker=marker, alpha=alpha, label=label)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_all_figures(
    dataset,
    forward,
    reverse,
    history: pd.DataFrame,
    output_dir,
    experiment: str,
    quality_sweep: pd.DataFrame | None = None,
) -> list[Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    made = []

    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d"); _sphere(ax); _scatter(ax, dataset); ax.set_title("Original Dataset")
    made.append(output / "01_dataset_bloch.png"); _save(fig, made[-1])

    steps = forward.steps; columns = min(4, len(steps)); rows = int(np.ceil(len(steps) / columns))
    fig = plt.figure(figsize=(4 * columns, 4 * rows))
    for i, t in enumerate(steps, 1):
        ax = fig.add_subplot(rows, columns, i, projection="3d"); _sphere(ax); _scatter(ax, forward.get_state(t)); ax.set_title(f"t={t}")
    made.append(output / "02_forward_diffusion_bloch.png"); _save(fig, made[-1])

    radii = torch.stack([torch.linalg.vector_norm(density_to_bloch(forward.get_state(t)), dim=1) for t in steps])
    fig, ax = plt.subplots(figsize=(7, 4)); x=np.asarray(steps); mean=radii.mean(1).cpu().numpy(); std=radii.std(1).cpu().numpy(); ax.plot(x,mean,"o-"); ax.fill_between(x,mean-std,mean+std,alpha=.25); ax.set(xlabel="Diffusion step",ylabel="Mean Bloch radius")
    made.append(output / "03_bloch_radius_vs_step.png"); _save(fig, made[-1])

    purities = torch.stack([purity(forward.get_state(t)) for t in steps])
    fig, ax = plt.subplots(figsize=(7, 4)); mean=purities.mean(1).cpu().numpy(); std=purities.std(1).cpu().numpy(); ax.plot(x,mean,"s-"); ax.fill_between(x,mean-std,mean+std,alpha=.25); ax.set(xlabel="Diffusion step",ylabel="Average purity")
    made.append(output / "04_purity_vs_step.png"); _save(fig, made[-1])

    fig, ax = plt.subplots(figsize=(7, 4));
    for t, group in history.groupby("step"): ax.plot(group["global_epoch"],group["total_loss"],label=f"step {t}")
    ax.set(xlabel="Training update",ylabel="Loss"); ax.legend()
    made.append(output / "05_training_loss.png"); _save(fig, made[-1])

    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(history["global_epoch"],history["superfidelity"]); ax.set(xlabel="Training update",ylabel="Mean superfidelity")
    made.append(output / "06_training_fidelity.png"); _save(fig, made[-1])

    fig = plt.figure(figsize=(4 * columns, 4 * rows))
    for i, t in enumerate(sorted(reverse.steps, reverse=True), 1):
        ax=fig.add_subplot(rows,columns,i,projection="3d"); _sphere(ax); _scatter(ax,reverse.get_state(t),marker="^"); ax.set_title(f"t={t}")
    made.append(output / "07_reverse_diffusion_bloch.png"); _save(fig,made[-1])

    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,projection="3d"); _sphere(ax)
    count=min(5,len(dataset))
    for i in range(count):
        f=np.stack([density_to_bloch(forward.get_state(t)[i]).cpu().numpy() for t in steps]); r=np.stack([density_to_bloch(reverse.get_state(t)[i]).detach().cpu().numpy() for t in sorted(reverse.steps,reverse=True)])
        ax.plot(f[:,0],f[:,1],f[:,2],"--o",alpha=.7,label="Forward" if i==0 else None); ax.plot(r[:,0],r[:,1],r[:,2],"-^",alpha=.7,label="Reverse" if i==0 else None)
    ax.legend(); made.append(output/"08_forward_reverse_trajectory.png"); _save(fig,made[-1])

    generated=reverse.get_state(0).detach()
    fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(231,projection="3d"); _sphere(ax); _scatter(ax,dataset); ax.set_title("Ground Truth"); ax=fig.add_subplot(232,projection="3d"); _sphere(ax); _scatter(ax,generated,marker="^"); ax.set_title("Generated")
    truth=density_to_bloch(dataset).cpu().numpy(); gen=density_to_bloch(generated).cpu().numpy()
    for idx,(a,b) in enumerate(((0,1),(0,2),(1,2)),4):
        ax=fig.add_subplot(2,3,idx); ax.scatter(truth[:,a],truth[:,b],s=10,label="Truth"); ax.scatter(gen[:,a],gen[:,b],s=10,marker="x",label="Generated"); ax.set(xlabel="XYZ"[a],ylabel="XYZ"[b]); ax.legend()
    made.append(output/"09_ground_truth_vs_generated.png"); _save(fig,made[-1])

    if quality_sweep is None or len(quality_sweep) < 2:
        raise ValueError("Figure 10 requires an aggregate quality sweep with at least two T values")
    sweep = quality_sweep.sort_values("T")
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(sweep["T"],sweep["superfidelity"],"D-"); ax.set(xlabel="Number of reverse steps T",ylabel="Nearest superfidelity",title="Additional smoke experiment (not paper-scale)")
    made.append(output/"10_quality_vs_steps.png"); _save(fig,made[-1])

    selected=sorted(set((0,max(steps)//2,max(steps))))
    matrices=[(f"forward {t}",forward.get_state(t)[0]) for t in selected]+[(f"reverse {t}",reverse.get_state(t)[0].detach()) for t in reversed(selected)]
    fig,axes=plt.subplots(2,len(matrices),figsize=(3*len(matrices),6))
    for j,(label,matrix) in enumerate(matrices):
        axes[0,j].imshow(matrix.real.cpu(),cmap="RdBu",vmin=-1,vmax=1); axes[0,j].set_title(label+" Re")
        axes[1,j].imshow(matrix.imag.cpu(),cmap="RdBu",vmin=-1,vmax=1); axes[1,j].set_title(label+" Im")
    made.append(output/"11_density_matrix_heatmap.png"); _save(fig,made[-1])

    fig,ax=plt.subplots(figsize=(7,4))
    # MPS has no Hermitian eigensolver; this is detached diagnostic work.
    fe=torch.stack([torch.linalg.eigvalsh(forward.get_state(t).detach().cpu().to(torch.complex128)).mean(0) for t in steps]).numpy(); re=torch.stack([torch.linalg.eigvalsh(reverse.get_state(t).detach().cpu().to(torch.complex128)).mean(0) for t in steps]).numpy()
    for i in range(fe.shape[1]): ax.plot(steps,fe[:,i],"--o",label=f"Forward λ{i}"); ax.plot(steps,re[:,i],"-^",label=f"Reverse λ{i}")
    ax.set(xlabel="Step",ylabel="Mean eigenvalue"); ax.legend(ncol=2)
    made.append(output/"12_eigenvalue_evolution.png"); _save(fig,made[-1])
    return made
