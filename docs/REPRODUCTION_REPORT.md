# Original MSQuDDPM Reproduction Report

## Scope

The implementation reproduces the complete original 1-qubit pipeline and trajectory extraction. Few-Step student/distillation is deliberately excluded. Validation artifacts produced from `smoke_*` configs are engineering smoke tests, not Table-I reproduction claims.

| Item | Paper | Our reproduction | Match |
|---|---|---|---|
| Cluster dataset | `epsilon=0.08`, `q∈[0,.01)` | Literal v2 definition | ✓ |
| Circle dataset | `RY(theta)|0>`, X–Z circle | Literal definition | ✓ |
| Forward | One global depolarizing map/step | One global map/step | ✓ |
| Schedule | cosine / cosine-square; offset omitted | offset `0.001`, clipping from official code | △ ASSUMPTION |
| System qubits | 1 | 1 | ✓ |
| Ancilla | Cluster 2 Haar+zero; circle 2 zero | Paper configs match; smoke uses 1 | ✓/△ |
| PQC | RX/RY + neighboring CZ | Same, independent block per step | ✓ |
| Depth | Cluster 4; circle 8 | Paper configs match; smoke uses 1 | ✓/△ |
| Measurement | Z projective, branch collected | Conditional sampled branch, label discarded | ✓ |
| Loss | Superfidelity Wasserstein | Same; smoke uses faster MMD | ✓/△ |
| Optimizer | Adam + exponential decay; factor/cadence omitted | Adam, `gamma=1` (no effective decay) | △ ASSUMPTION |
| Quantum simulator | TensorCircuit with PyTorch backend | Native PyTorch density-matrix algebra | △ equivalent circuit, framework difference |
| LR/epochs | Not reported per task | Official CLI defaults in paper configs | △ ASSUMPTION |
| Result | Table I values in Step-1 report | Not yet paper-scale executed | Not claimed |

## Paper–code corrections

1. Official multi-qubit forward code repeats the global channel `n` times; this implementation applies paper Eq. (1) once.
2. Paper v2 cluster width `0.08` is used instead of official training script `0.04`.
3. Checkpoint, generation, deterministic seeds, validation, CLI, and trajectory persistence are implemented rather than left manual.
4. Empty global-loss return and unsupported `--p_limit` defects are absent.

## Output contract

- Checkpoints: `outputs/checkpoints/*.pt`
- History: `outputs/histories/*.csv`
- Metrics: `outputs/metrics/*.csv`
- Forward/reverse trajectories: both `.pt` and `.npz`
- Teacher trajectory: one `.npz` with distinct forward/reverse IDs, `paired=false`, and both independent ensemble chains
- Figures: all 12 required names per experiment

## Apple Silicon MPS execution

Apple MPS is supported as a hybrid engineering execution backend. Device selection order is CUDA → MPS → CPU. MPS uses `float32/complex64`; CPU/CUDA retain `float64/complex128`. Circuit evolution, differentiable loss values, and parameter gradients run on MPS. Reproducible measurement sampling and POT's detached transport-plan solve are small CPU control operations; detached eigendecomposition diagnostics also run on CPU because MPS lacks complex Hermitian `eigh/eigvalsh`. This backend/precision change is marked △ rather than a paper-method change: circuit/channel/loss definitions are unchanged, but stochastic numerical results need not match CPU exactly.

Validated MPS artifacts use `smoke_clustered_mps` and `smoke_circular_mps`; no paper-scale MPS claim is made.

## Interpretation limits

Smoke runs use smaller `T`, depth, batch, and epochs. They validate code paths, physicality, stochastic reverse blocks, persistence, metrics, and visualization only. Figure 10 uses a separate two-point T sweep explicitly labeled an additional smoke experiment. Cluster outputs include the paper metric `F_data_0` and `F_gen_0`; generic nearest-state fidelity remains diagnostic. Table-I comparison requires running `configs/1q_clustered.yaml` and `configs/1q_circular.yaml`, ideally over multiple recorded seeds; the paper does not disclose the error-bar protocol or all optimizer settings.
