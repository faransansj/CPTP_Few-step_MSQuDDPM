# Validation Report — Steps 2–8

Date: 2026-08-13. Environment: Python 3.11.15, PyTorch 2.13.0, CPU smoke configurations. Command spellings were normalized to the locked `uv` workflow on 2026-08-14; `uv sync`, the full tests, and clustered smoke were rerun, while the other outcome rows retain the original validated evidence.

## Commands and outcomes

| Command | Exit | Result |
|---|---:|---|
| `uv sync --extra test --locked` | 0 | Python 3.11 environment and locked dependencies installed |
| `uv run --locked pytest -q` | 0 | 9 passed, 2 hardware-only tests skipped |
| `uv run --locked python scripts/train.py --config configs/smoke_clustered.yaml` | 0 | 2 reverse blocks × 12 epochs, rerun after migration |
| `uv run --locked python scripts/train.py --config configs/smoke_circular.yaml` | 0 | 2 reverse blocks × 12 epochs |
| `uv run --locked python scripts/quality_sweep.py --config ... --steps 1 2` (both) | 0 | two-point additional smoke sweep CSVs |
| `uv run --locked python scripts/evaluate.py --checkpoint ...` (both) | 0 | finite metrics, complete valid trajectories |
| `uv run --locked python scripts/inspect_trajectory.py ...` (four directions) | 0 | forward/reverse step 0–2 inspected for both datasets |
| `uv run --locked python scripts/visualize.py --experiment ...` (both) | 0 | 12 PNGs each, Figure 10 consumes sweep CSV |

Development/review exposed and fixed detached categorical probability NaNs, differentiation through complex eigendecomposition, RX/RY action order, frozen measurement draws, and incomplete teacher/metric contracts. One corrected training invocation returned shell exit 1 only because `tee` targeted a deleted output directory; the training itself finished. The directory was created and the identical deterministic clustered run was rerun successfully.

## Numerical evidence

### Physical forward process

Clustered mean Bloch radius: `0.996110 → 0.497275 → 0.000000`; mean purity: `0.996121 → 0.623642 → 0.500000`. All matrices passed Hermitian/trace/PSD/purity checks.

Both final forward and reverse trajectories for clustered and circular report `valid=True` at every step. `.pt` versus `.npz` round-trip maximum absolute difference is `0.0`.

### Training

Corrected runs use fixed per-stage Haar inputs and advancing projective-measurement RNGs. Losses are therefore stochastic and need not decrease monotonically.

- Cluster step 2: `0.119073 → 0.120959`, minimum `0.100941`; step 1: `0.549477 → 0.645476`, minimum `0.497061`.
- Circle step 2: `0.047184 → 0.029099`; step 1: `0.243997 → 0.153914`, minimum `0.152253`.

This validates finite optimization and parameter updates, not convergence quality or Table-I reproduction.

### Final smoke metrics

| Experiment | Nearest superfidelity | MMD | Wasserstein | Generated radius |
|---|---:|---:|---:|---:|
| clustered | 0.5920 | 0.7041 | 0.4303 | 0.5634 |
| circular | 0.8692 | 0.1456 | 0.3146 | 0.7333 |

Cluster paper metric: `F_data_0=0.97698`, `F_gen_0=0.48024`. Circular diagnostic overlaps are `0.52008` and `0.45303` respectively.

Smoke quality is intentionally bounded by `T=2`, depth 2, 10 states, 12 epochs.

## Trajectory evidence

Each teacher file contains:

```text
forward_sample_id, reverse_sample_id, sample_id, paired=false, steps,
rho_0, rho_1, rho_2,
reverse_rho_0, reverse_rho_1, reverse_rho_2
```

Forward and reverse rows are independent ensemble paths; `sample_id` is a legacy positional index and does not assert pairing. All teacher matrices passed validation: worst Hermiticity residual `2.02e-16`, trace residual `2.23e-16`, minimum eigenvalue `3.06e-4`. Production paper configs yield the same schema through `rho_6`.

## Figure evidence

Both smoke experiments contain exactly 12 nonempty PNGs, dimensions `960×640` through `2880×960`. Figure 10 uses actual additional-smoke sweep points: clustered `(T=1, 0.6160)`, `(T=2, 0.5585)`; circular `(T=1, 0.8266)`, `(T=2, 0.8147)`. These two-point stochastic smoke trends are diagnostics, not evidence that quality must increase with T.

## Apple Silicon MPS validation

PyTorch 2.13.0 reported MPS built/available. Core tensors were observed on `mps:0`: model parameters `float32`, datasets and forward/reverse states `complex64`; the conditional MPS test performs a real optimizer update and confirms an MPS-resident gradient. Full suite: `9 passed, 1 CUDA-only skipped`.

Actual smoke runs completed without `PYTORCH_ENABLE_MPS_FALLBACK`:

| Experiment | Train wall time | Superfidelity | MMD | Wasserstein | Device/dtypes |
|---|---:|---:|---:|---:|---|
| clustered MPS | 5.91 s | 0.5173 | 0.7852 | 0.4856 | `mps`, `float32/complex64` |
| circular MPS | 4.29 s | 0.8316 | 0.1275 | 0.2443 | `mps`, `float32/complex64` |

Both evaluation, forward/reverse inspection, two-point quality sweeps, and all 12 figures completed. Worst physical residuals across saved MPS trajectories: Hermiticity `1.08e-7`, trace `1.92e-7`, minimum eigenvalue `3.46e-4`; all pass the documented low-precision `2e-5` tolerance. CPU metrics above are comparison diagnostics only because stochastic measurement paths and numeric precision differ.

MPS does not support `float64/complex128` or complex `eigh/eigvalsh`. The implementation therefore selects `float32/complex64` centrally for MPS. Circuit evolution, differentiable costs, and parameter gradients remain on MPS. Reproducible categorical measurement draws use CPU probabilities/outcome indices, and Wasserstein uses POT on CPU to solve a detached transport plan; the chosen cost is then weighted and differentiated on MPS. Detached validation, state-fidelity, trace-distance, and eigenvalue-plot inputs also use CPU `complex128`. Thus the validated path is intentionally hybrid rather than fully GPU-native.

## Paper-scale follow-up

Paper-scale MPS configs subsequently completed for seeds 7, 42, and 123 on both datasets, and all trajectories passed physical validation. Those runs were later invalidated as reproduction evidence because the circuit alternated CZ edges instead of applying every neighboring CZ in every layer. Their historical values and limitations remain in [`PAPER_SCALE_MPS_RESULTS.md`](PAPER_SCALE_MPS_RESULTS.md); they support only execution and physicality of the obsolete circuit. Corrected CPU diagnostics are recorded in [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md), and MPS revalidation is out of scope.
