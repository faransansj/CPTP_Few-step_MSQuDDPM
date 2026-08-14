import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from msquddpm.datasets import circular_states, clustered_states
from msquddpm.experiment import load_experiment, train_experiment
from msquddpm.forward_diffusion import ForwardDiffusion, noise_schedule
from msquddpm.losses import mmd_loss, superfidelity, wasserstein_loss
from msquddpm.metrics import nearest_metrics, zero_state_overlap
from msquddpm.precision import precision_for
from msquddpm.reverse_model import ReverseMSQuDDPM, _cz, _expand_gate, _rx, _ry
from msquddpm.states import density_to_bloch, purity, validate_density_matrix
from msquddpm.trajectory import Trajectory, load_trajectory, save_teacher_trajectory, save_trajectory
from msquddpm.visualization import generate_all_figures
from msquddpm.trainer import train_greedy
from msquddpm.utils import get_device


def test_datasets_and_forward_are_physical_and_mix():
    for states in (clustered_states(12, seed=1), circular_states(12, seed=1)):
        assert validate_density_matrix(states)["valid"]
        trajectory = ForwardDiffusion(4, "cosine").diffuse(states)
        assert trajectory.steps == list(range(5))
        radii = [torch.linalg.vector_norm(density_to_bloch(trajectory.get_state(t)), dim=1).mean() for t in trajectory.steps]
        purities = [purity(trajectory.get_state(t)).mean() for t in trajectory.steps]
        assert all(a >= b - 1e-9 for a, b in zip(radii, radii[1:]))
        assert all(a >= b - 1e-9 for a, b in zip(purities, purities[1:]))
        assert radii[-1] < 1e-6


def test_pqc_order_is_rx_then_ry_then_all_neighboring_cz():
    model = ReverseMSQuDDPM(steps=1, n_ancilla=2, depth=1, ancilla="zero", seed=3)
    with torch.no_grad():
        model.theta.zero_()
        model.theta[0, 0, 0] = torch.tensor([0.71, -0.43])
    rx = _expand_gate(_rx(model.theta[0, 0, 0, 0]), 0, 3)
    ry = _expand_gate(_ry(model.theta[0, 0, 0, 1]), 0, 3)
    expected = _cz(3, 1, model.theta.device) @ _cz(3, 0, model.theta.device) @ ry @ rx
    wrong_order = _cz(3, 1, model.theta.device) @ _cz(3, 0, model.theta.device) @ rx @ ry
    missing_neighbor = _cz(3, 0, model.theta.device) @ ry @ rx
    assert torch.allclose(model._unitary(1), expected)
    assert not torch.allclose(expected, wrong_order)
    assert not torch.allclose(expected, missing_neighbor)


def test_losses_reverse_api_and_trajectory_roundtrip(tmp_path):
    states = clustered_states(5, seed=3)
    assert torch.isfinite(superfidelity(states[0], states[1]))
    assert abs(float(mmd_loss(states, states))) < 1e-10
    assert abs(float(wasserstein_loss(states, states))) < 1e-10
    model = ReverseMSQuDDPM(steps=2, n_ancilla=1, depth=1, seed=3)
    mixed = torch.eye(2, dtype=torch.complex128)[None].repeat(5, 1, 1) / 2
    step = model.reverse_step(mixed, t=2)
    assert validate_density_matrix(step, atol=1e-6)["valid"]
    trajectory = model.generate(mixed, return_trajectory=True)
    assert trajectory.steps == [0, 1, 2]
    assert torch.allclose(model.get_state(1), trajectory.get_state(1))
    for suffix in (".pt", ".npz"):
        path = tmp_path / f"trajectory{suffix}"
        save_trajectory(trajectory, path)
        loaded = load_trajectory(path)
        for t in trajectory.steps:
            assert torch.allclose(trajectory.get_state(t), loaded.get_state(t))


def test_training_sampling_semantics_and_decay_cadence():
    states = clustered_states(4, seed=12)
    forward = ForwardDiffusion(2).diffuse(states)

    calls = {}
    for semantics in ("official", "replay"):
        model = ReverseMSQuDDPM(2, n_ancilla=1, depth=1, ancilla="zero", seed=12)
        with patch.object(model, "reverse_step", wraps=model.reverse_step) as reverse_step:
            train_greedy(model, forward, epochs=2, learning_rate=0.01, loss_name="mmd", sampling_semantics=semantics)
        calls[semantics] = reverse_step.call_count
    assert calls == {"official": 5, "replay": 6}

    model = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="zero", seed=12)
    one_step = Trajectory({0: forward.get_state(0), 1: forward.get_state(1)}, "forward")
    history = train_greedy(model, one_step, epochs=8, learning_rate=0.01, loss_name="mmd", gamma=0.5, lr_decay_count=4).history
    assert history["learning_rate"].tolist() == pytest.approx([0.01, 0.005, 0.005, 0.0025, 0.0025, 0.00125, 0.00125, 0.00125])

    model = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="zero", seed=12)
    history = train_greedy(model, one_step, epochs=9, learning_rate=0.01, loss_name="mmd", gamma=0.5, lr_decay_count=2).history
    assert history["learning_rate"].tolist() == pytest.approx([0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])

    with pytest.raises(ValueError, match="sampling semantics"):
        train_greedy(model, one_step, epochs=1, learning_rate=0.01, sampling_semantics="unknown")
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_greedy(model, one_step, epochs=0, learning_rate=0.01)
    with pytest.raises(ValueError, match="lr_decay_count"):
        train_greedy(model, one_step, epochs=1, learning_rate=0.01, lr_decay_count=2)


def test_validation_does_not_project_or_modify_reverse_output():
    a = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="zero", seed=8)
    b = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="zero", seed=8)
    b.load_state_dict(a.state_dict())
    mixed = torch.eye(2, dtype=torch.complex128)[None].repeat(4, 1, 1) / 2
    raw = a.reverse_step(mixed, 1, validate=False, measurement_generator=torch.Generator().manual_seed(99))
    checked = b.reverse_step(mixed, 1, validate=True, measurement_generator=torch.Generator().manual_seed(99))
    assert torch.equal(raw, checked)


def test_teacher_schema_is_unpaired_and_validated(tmp_path):
    states = clustered_states(4, seed=4)
    forward = ForwardDiffusion(2).diffuse(states)
    reverse = ReverseMSQuDDPM(2, n_ancilla=1, depth=1, ancilla="zero", seed=4).generate(
        torch.eye(2, dtype=torch.complex128)[None].repeat(4, 1, 1) / 2, True
    )
    path = save_teacher_trajectory(forward, reverse, tmp_path / "nested" / "teacher.npz")
    with np.load(path, allow_pickle=False) as data:
        assert {"forward_sample_id", "reverse_sample_id", "sample_id", "paired", "steps"} <= set(data.files)
        assert not bool(data["paired"])
        assert data["steps"].tolist() == [0, 1, 2]
        for t in data["steps"]:
            assert data[f"rho_{t}"].shape == data[f"reverse_rho_{t}"].shape == (4, 2, 2)
    bad = Trajectory({0: states[:3], 1: states, 2: states}, "reverse")
    with pytest.raises(ValueError, match="shape"):
        save_teacher_trajectory(forward, bad, tmp_path / "bad.npz")


def _tiny_config(tmp_path):
    return {
        "experiment": "tiny",
        "dataset": "clustered",
        "seed": 5,
        "dataset_size": 4,
        "T": 1,
        "schedule": "cosine",
        "schedule_offset": 0.001,
        "n_ancilla": 1,
        "depth": 1,
        "ancilla": "haar",
        "loss": "mmd",
        "epochs": 2,
        "learning_rate": 0.01,
        "gamma": 1.0,
        "init": "normal",
        "device": "cpu",
        "output_root": str(tmp_path),
    }


def test_end_to_end_training_checkpoint_metrics_and_figures(tmp_path):
    result = train_experiment(_tiny_config(tmp_path))
    assert len(result["history"]) == 2 and np.isfinite(result["history"]["total_loss"]).all()
    model, config, target = load_experiment(result["checkpoint"], "cpu")
    assert config["experiment"] == "tiny" and target.shape == (4, 2, 2)
    assert {"F_data_0", "F_gen_0"} <= result["metrics"].keys()
    assert result["metrics"]["F_data_0"] == pytest.approx(zero_state_overlap(target))
    sweep = pd.DataFrame({"T": [1, 2], "superfidelity": [0.5, 0.6]})
    figures = generate_all_figures(
        target,
        result["forward"],
        result["reverse"],
        result["history"],
        tmp_path / "figures" / "tiny",
        "tiny",
        sweep,
    )
    expected = {f"{i:02d}_{name}.png" for i, name in enumerate((
        "dataset_bloch", "forward_diffusion_bloch", "bloch_radius_vs_step", "purity_vs_step",
        "training_loss", "training_fidelity", "reverse_diffusion_bloch", "forward_reverse_trajectory",
        "ground_truth_vs_generated", "quality_vs_steps", "density_matrix_heatmap", "eigenvalue_evolution"
    ), 1)}
    assert {p.name for p in figures} == expected
    assert all(p.stat().st_size > 1000 for p in figures)


def test_inspect_cli_requires_exactly_one_source():
    script = Path(__file__).parents[1] / "scripts" / "inspect_trajectory.py"
    none = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    both = subprocess.run([sys.executable, str(script), "--experiment", "x", "--trajectory", "x.pt"], capture_output=True, text=True)
    assert none.returncode == 2 and both.returncode == 2


def test_schedule_terminal_noise_and_metrics():
    for kind in ("cosine", "sq_cosine"):
        beta = noise_schedule(6, kind)
        assert beta.shape == (6,) and torch.all((beta >= 0) & (beta <= 1))
        assert beta[-1] == 1
    states = clustered_states(4, seed=2)
    metrics = nearest_metrics(states, states)
    assert metrics["F_data_0"] == pytest.approx(metrics["F_gen_0"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_precision_forward_reverse_and_optimizer():
    device = torch.device("mps")
    assert get_device("auto").type == ("cuda" if torch.cuda.is_available() else "mps")
    precision = precision_for(device)
    states = clustered_states(4, seed=9, device=device)
    forward = ForwardDiffusion(1).diffuse(states)
    model = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="haar", seed=9, device=device)
    before = model.theta.detach().cpu().clone()
    # Exercise the paper-scale loss path too: POT solves the detached transport
    # plan on CPU, while the selected cost and parameter gradients remain on MPS.
    result = train_greedy(
        model,
        forward,
        epochs=1,
        learning_rate=0.01,
        loss_name="wasserstein",
        lr_decay_count=1,
    )
    mixed = torch.eye(2, dtype=precision.complex, device=device)[None].repeat(4, 1, 1) / 2
    reverse = model.generate(mixed, True)
    assert states.device.type == forward.get_state(1).device.type == reverse.get_state(0).device.type == "mps"
    assert states.dtype == forward.get_state(1).dtype == reverse.get_state(0).dtype == torch.complex64
    assert model.theta.device.type == "mps" and model.theta.dtype == torch.float32
    assert model.theta.grad is not None and model.theta.grad.device.type == "mps"
    assert not torch.equal(before, model.theta.detach().cpu())
    assert np.isfinite(result.history["total_loss"]).all()
    assert validate_density_matrix(reverse.get_state(0))["valid"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reverse_smoke():
    model = ReverseMSQuDDPM(1, n_ancilla=1, depth=1, ancilla="zero", seed=1).cuda()
    mixed = torch.eye(2, dtype=torch.complex128, device="cuda")[None] / 2
    assert validate_density_matrix(model.reverse_step(mixed, 1).cpu())["valid"]
