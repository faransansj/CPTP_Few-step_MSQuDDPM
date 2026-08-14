# Original MSQuDDPM Reproduction

PyTorch density-matrix reproduction of *Mixed-State Quantum Denoising Diffusion Probabilistic Model* ([arXiv:2411.17608v2](https://arxiv.org/abs/2411.17608v2)), based primarily on the authors' [`gkwun/msquddpm`](https://github.com/gkwun/msquddpm) implementation at commit `158df6e9474aca6a9ab00d01b60fe4d65cc093ba`. The same RX/RY/CZ density-matrix circuit is implemented directly in PyTorch rather than through TensorCircuit, avoiding an extra runtime layer while retaining autograd and the official framework's PyTorch backend semantics.

This repository implements only the original MSQuDDPM teacher baseline. It does **not** implement Few-Step students or distillation.

The clustered failure was traced to an incorrect alternating-CZ implementation: the official circuit applies every neighboring CZ in every layer. After correction, a diagnostic CPU three-seed run reached `F_gen,0=0.96269±0.02010`, passing the predeclared mean target of `0.95`. The earlier MPS results used the incorrect circuit and are retained only as invalidated historical evidence, including the [seed-42 figure gallery](docs/figures/paper_scale_mps/README.md). The teacher baseline remains unfrozen until clean CPU artifacts and provenance are recorded; see [`docs/REPRODUCTION_REPORT.md`](docs/REPRODUCTION_REPORT.md) and [`docs/NEXT_PHASE_ROADMAP.md`](docs/NEXT_PHASE_ROADMAP.md).

## Installation

```bash
uv sync --extra test --locked
uv run --locked pytest -q
```

`uv` 0.11 or newer reads `.python-version`, creates and manages `.venv`, and installs the exact versions in `uv.lock`; manual activation and `pip` are unnecessary. The locked Torch wheels support Apple Silicon on macOS 14+, Linux with glibc 2.28+ on x86-64/AArch64, and Windows x86-64. Intel macOS is not supported by this lock.

`device: auto` selects CUDA, then Apple MPS, then CPU. CPU/CUDA use `float64/complex128`; MPS uses `float32/complex64`. Circuit evolution and differentiable losses run on the selected accelerator. Reproducible categorical measurement sampling and POT's detached optimal-transport-plan solve are CPU control operations; gradients through the selected Wasserstein cost remain on the accelerator. MPS additionally sends detached eigendecomposition diagnostics to CPU because PyTorch MPS lacks complex Hermitian eigensolvers. MPS validation uses `atol=2e-5` versus `1e-7` at research precision.

## CUDA server quick start

The project does not install a system-wide CUDA toolkit. On Linux, the locked PyTorch package includes its user-space CUDA 13 libraries. Use the lock when the server driver supports it; otherwise apply the explicit CUDA 12.8 deviation below and revalidate that environment.

```bash
git clone https://github.com/faransansj/CPTP_Few-step_MSQuDDPM.git
cd CPTP_Few-step_MSQuDDPM

# Create the locked Python 3.11 environment.
uv sync --extra test --locked

# Optional CUDA 12.8 deviation: replace the locked Torch stack with a pinned wheel.
# Use `uv run --no-sync` afterward so the project lock is not reapplied.
uv pip install --reinstall 'torch==2.11.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
```

Confirm that the GPU is really visible:

```bash
nvidia-smi
uv run --no-sync python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available()
PY
```

Run tests and a CUDA smoke training before paper-scale jobs:

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/train.py --config configs/smoke_clustered_cuda.yaml
uv run --no-sync python scripts/train.py --config configs/smoke_circular_cuda.yaml
uv run --no-sync python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/smoke_clustered_cuda.pt --device cuda
```

Paper-scale configs use `device: auto`, which selects CUDA on a CUDA server:

```bash
uv run --no-sync python scripts/train.py --config configs/1q_clustered.yaml
uv run --no-sync python scripts/train.py --config configs/1q_circular.yaml
```

For an unattended server job, keep stdout and timing information:

```bash
mkdir -p outputs/logs
nohup /usr/bin/time -v uv run --no-sync python scripts/train.py --config configs/1q_clustered.yaml \
  > outputs/logs/1q_clustered_cuda.log 2>&1 &
echo $! > outputs/logs/1q_clustered_cuda.pid
```

The CUDA 12.8 override is intentionally outside `uv.lock` and may replace related Torch dependencies. To restore the baseline exactly, delete `.venv` and rerun `uv sync --extra test --locked`.

Generated checkpoints, trajectories, figures, metrics, histories, and logs live under `outputs/` and are intentionally git-ignored. Copy them separately from the server. CUDA currently uses `float64/complex128`; verify that the selected GPU supports efficient FP64 if runtime matters. The POT transport-plan solve and measurement sampling remain CPU-assisted, so additional CPU cores and fast host-device transfers still help.

## Apple Silicon MPS

Apple Silicon smoke run:

```bash
uv run --locked python scripts/train.py --config configs/smoke_clustered_mps.yaml
uv run --locked python scripts/train.py --config configs/smoke_circular_mps.yaml
uv run --locked python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_clustered_mps.pt --device mps
uv run --locked python scripts/quality_sweep.py --config configs/smoke_clustered_mps.yaml --steps 1 2
uv run --locked python scripts/visualize.py --experiment smoke_clustered_mps
```

Known limit: this is a hybrid MPS execution path, not a claim that every auxiliary operation is GPU-native. MPS results are lower-precision stochastic diagnostics and are not expected to equal CPU metrics exactly. Checkpoint loading stages through CPU to avoid materializing float64 schedule metadata on MPS.

## Dataset

- Clustered: paper-v2 `|ψ> ∝ |0> + 0.08 c|1>`, complex-normal `c`, `q ~ U[0,0.01)`.
- Circular: literal paper `RY(θ)|0>`, `θ ~ U[0,2π)`, `q ~ U[0,0.04)`.
- Every generated state is checked for Hermiticity, trace one, PSD, and valid purity.

## Training

Paper-scale configuration (expensive; Table-I result is not claimed until run):

```bash
uv run --locked python scripts/train.py --config configs/1q_clustered.yaml
uv run --locked python scripts/train.py --config configs/1q_circular.yaml
```

CPU smoke reproduction:

```bash
uv run --locked python scripts/train.py --config configs/smoke_clustered.yaml
uv run --locked python scripts/train.py --config configs/smoke_circular.yaml
```

Training follows the paper's greedy `T → 1` process. Each RX/RY/CZ block is separately addressable with `model.reverse_step(rho, t)`. Ancilla Z measurements sample conditional post-measurement states; outcomes are not postselected or retained as labels.

## Evaluation

```bash
uv run --locked python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_clustered.pt
```

CSV output contains nearest-state fidelity/superfidelity, trace distance, MMD, Wasserstein, purity error, Bloch radii, and the clustered paper metric `F_data_0`/`F_gen_0 = mean(<0|rho|0>)`. Distributional MMD/Wasserstein are primary; nearest-state metrics are diagnostics.

## Visualization

```bash
uv run --locked python scripts/visualize.py --experiment smoke_clustered
```

First run the bounded **additional smoke experiment** (not a paper/Table-I sweep), then render figures:

```bash
uv run --locked python scripts/quality_sweep.py --config configs/smoke_clustered.yaml --steps 1 2
uv run --locked python scripts/visualize.py --experiment smoke_clustered
```

This creates all required files `01_dataset_bloch.png` through `12_eigenvalue_evolution.png` under `outputs/figures/<experiment>/`. Figure 10 consumes `outputs/metrics/<experiment>_quality_vs_steps.csv` and requires at least two actual trained T values.

## Trajectory inspection

```bash
uv run --locked python scripts/inspect_trajectory.py --experiment smoke_clustered --direction forward
uv run --locked python scripts/inspect_trajectory.py --experiment smoke_clustered --direction reverse
```

API:

```python
trajectory = model.generate(rho_T, return_trajectory=True)
rho_4 = trajectory.get_state(4)
rho_4_again = model.get_state(4)
rho_next = model.reverse_step(rho, t=8)
save_trajectory(trajectory, "trajectory.pt")  # also .npz
trajectory = load_trajectory("trajectory.pt")
```

Teacher `.npz` files expose `forward_sample_id`, `reverse_sample_id`, legacy positional `sample_id`, `paired=false`, `rho_0...rho_T`, and `reverse_rho_0...reverse_rho_T`. Forward and reverse rows are independent ensemble paths; equal row indices do not imply coupled samples.

## Teacher artifact contract and canonical baselines

Frozen teacher baselines use the strict Teacher artifact schema v1 (steps 0–6, `(batch, 2, 2)` complex128, tolerance 1e-8, both forward and reverse chains, unpickled scalar/string metadata with config/checkpoint hashes). See [`docs/TEACHER_ARTIFACT_SCHEMA.md`](docs/TEACHER_ARTIFACT_SCHEMA.md) for the full schema and policies.

Canonical CPU baseline configs live in `configs/baselines/` (`clustered_seed{7,42,123}.yaml`, `circular_seed{7,42,123}.yaml`). They require a clean Git worktree and write an immutable `manifest.json` provenance record; reruns fail rather than overwrite. Clustered acceptance is aggregate-only: mean `F_gen,0 ≥ 0.95` across seeds 7/42/123, never per-seed. Circular configs carry an unresolved acceptance-metric TODO and are blocked by preflight.

```bash
uv run --locked python scripts/train.py --config configs/baselines/clustered_seed7.yaml
uv run --locked python scripts/validate_teacher.py \
  outputs/baselines/clustered_seed7/trajectories/clustered_seed7_teacher.npz \
  --config outputs/baselines/clustered_seed7/config.resolved.yaml \
  --checkpoint outputs/baselines/clustered_seed7/checkpoints/clustered_seed7.pt
```

## Reproduction procedure

1. Read [`docs/STEP1_PAPER_AND_OFFICIAL_CODE_ANALYSIS.md`](docs/STEP1_PAPER_AND_OFFICIAL_CODE_ANALYSIS.md).
2. Run tests.
3. Run both smoke configs and inspect numerical/figure outputs.
4. Run paper configs with recorded hardware/runtime and multiple seeds.
5. Compare with Table I using [`docs/REPRODUCTION_REPORT.md`](docs/REPRODUCTION_REPORT.md); never label smoke output paper reproduction.

## Documented assumptions

- `epsilon=0.001` and beta clipping follow official code because the paper omits the offset value.
- Paper-scale LR/epochs use official CLI defaults but are not paper-attested per-task settings. `gamma=1` means Adam with **no effective learning-rate decay**; the paper's decay factor/cadence are unknown.
- Dataset interval parameters are sampled uniformly, following official code.
- The implementation fixes the official multi-qubit repeated-global-channel defect, default training crash, unsupported README option, and unusable generation parameter path.

## Progress

- [x] Step 1: paper/official-code analysis
- [x] Steps 2–8: modular implementation and CPU smoke validation
- [x] Clustered root cause isolated and corrected; diagnostic CPU three-seed mean passes target
- [ ] Clean CPU multi-seed baseline regeneration, provenance, and freeze
