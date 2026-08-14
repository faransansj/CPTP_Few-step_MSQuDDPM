from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from .precision import precision_for
from .losses import mmd_loss, pairwise_superfidelity, wasserstein_loss


@dataclass
class TrainResult:
    history: pd.DataFrame


def train_greedy(
    model,
    forward_trajectory,
    epochs: int,
    learning_rate: float,
    loss_name: str = "wasserstein",
    gamma: float = 1.0,
    log_every: int = 1,
    sampling_semantics: str = "official",
    lr_decay_count: int = 2,
) -> TrainResult:
    """Paper's T-to-1 greedy training; only the current block receives gradients."""
    if sampling_semantics not in {"official", "replay"}:
        raise ValueError(f"Unknown sampling semantics: {sampling_semantics}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 1 <= lr_decay_count <= epochs:
        raise ValueError("lr_decay_count must be in [1, epochs]")
    decay_interval = epochs // lr_decay_count
    rows: list[dict] = []
    batch = len(forward_trajectory.get_state(0))
    mixed = torch.eye(2, dtype=precision_for(model.theta.device).complex, device=model.theta.device)[None].repeat(batch, 1, 1) / 2
    loss_function = wasserstein_loss if loss_name == "wasserstein" else mmd_loss

    for t in range(model.steps, 0, -1):
        optimizer = torch.optim.Adam([model.theta], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        target = forward_trajectory.get_state(t - 1).to(model.theta.device)
        # The paper fixes stage Haar inputs during optimization, while projective
        # outcomes remain stochastic. These use deliberately separate RNG streams.
        ancilla_rng = torch.Generator().manual_seed(model.seed + 10_000 * t)
        fixed_ancillas = {
            stage_t: model._ancillas(batch, model.theta.device, ancilla_rng)
            for stage_t in range(model.steps, t - 1, -1)
        }
        measurement_rng = torch.Generator().manual_seed(model.seed + 20_000 * t)
        stage_input = mixed
        if sampling_semantics == "official":
            # Official code samples later trained blocks once before this stage's
            # epoch loop; only the current block's measurement is resampled.
            with torch.no_grad():
                for fixed_t in range(model.steps, t, -1):
                    stage_input = model.reverse_step(
                        stage_input,
                        fixed_t,
                        validate=False,
                        ancilla_states=fixed_ancillas[fixed_t],
                        measurement_generator=measurement_rng,
                    )
        for epoch in range(epochs):
            optimizer.zero_grad()
            current = stage_input
            if sampling_semantics == "replay":
                with torch.no_grad():
                    for fixed_t in range(model.steps, t, -1):
                        current = model.reverse_step(
                            current,
                            fixed_t,
                            validate=False,
                            ancilla_states=fixed_ancillas[fixed_t],
                            measurement_generator=measurement_rng,
                        )
            output = model.reverse_step(
                current.detach(),
                t,
                validate=False,
                ancilla_states=fixed_ancillas[t],
                measurement_generator=measurement_rng,
            )
            loss = loss_function(output, target)
            loss.backward()
            if model.theta.grad is not None:
                mask = torch.zeros_like(model.theta.grad)
                mask[t - 1] = 1
                model.theta.grad.mul_(mask)
            optimizer.step()
            # Mirrors the official integer-interval cadence exactly. Its CLI
            # count-minus-one description holds for the paper's 2001/2 setting.
            if gamma != 1.0 and (epoch + 1) % decay_interval == 0 and epoch + 2 < epochs:
                scheduler.step()
            if epoch % log_every == 0 or epoch == epochs - 1:
                rows.append(
                    {
                        "step": t,
                        "epoch": epoch,
                        "global_epoch": len(rows),
                        "total_loss": float(loss.detach()),
                        "step_loss": float(loss.detach()),
                        "superfidelity": float(pairwise_superfidelity(output.detach(), target).mean()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
    return TrainResult(pd.DataFrame(rows))
