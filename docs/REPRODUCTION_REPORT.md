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
| Result | Table I values in Step-1 report | Corrected clustered CPU diagnostic: `F_gen,0=0.96269±0.02010` over seeds 7/42/123 | Diagnostic pass; clean baseline pending |

## Paper–code corrections

1. Official multi-qubit forward code repeats the global channel `n` times; this implementation applies paper Eq. (1) once.
2. Paper v2 cluster width `0.08` is used instead of official training script `0.04`.
3. Checkpoint, generation, deterministic seeds, validation, CLI, and trajectory persistence are implemented rather than left manual.
4. Empty global-loss return and unsupported `--p_limit` defects are absent.
5. The reverse circuit applies every neighboring CZ in every layer, matching both even- and odd-pair loops in the official implementation.
6. Later trained blocks are sampled once before each stage epoch loop, and optional LR decay follows the official interval cadence.

## Clustered root-cause analysis

The failed implementation used `range(layer % 2, total - 1, 2)`, applying only one parity of CZ edges per layer. For the paper's three-qubit register, each layer therefore omitted one of `CZ(0,1)` or `CZ(1,2)`. The official TensorCircuit code executes both pair loops in every layer.

After changing the native unitary to apply all neighboring CZ edges, `scripts/check_tensorcircuit_parity.py` matched TensorCircuit with maximum absolute error `3.33e-16` against tolerance `1e-12`. Run it after `uv pip install --python .venv/bin/python tensorcircuit==0.11.0`. Official pretrained clustered parameters evaluated through the corrected circuit produced `F_gen,0` values `0.98714`, `0.98243`, and `0.98722` for seeds 7, 42, and 123.

Corrected paper-scale CPU diagnostics using official sampling semantics produced:

| Seed | `F_data,0` | `F_gen,0` |
|---:|---:|---:|
| 7 | 0.98301 | 0.98584 |
| 42 | 0.98390 | 0.94952 |
| 123 | 0.98683 | 0.95273 |
| Mean ± sample std | — | `0.96269±0.02010` |

The predeclared mean acceptance target `F_gen,0≥0.95` passes. A seed-42 replay-semantics control reached `0.94849`, showing that sampling semantics is secondary rather than the cause of the former `≈0.5` failure. Width, ancilla, initialization, and optimizer sweeps were cancelled because a single source-level cause was isolated and falsified directly.

These runs were diagnostic artifacts from a modified working tree. They establish the fix but are not immutable baseline artifacts; clean committed reruns and provenance remain required before tagging.

## Output contract

- Checkpoints: `outputs/checkpoints/*.pt`
- History: `outputs/histories/*.csv`
- Metrics: `outputs/metrics/*.csv`
- Forward/reverse trajectories: both `.pt` and `.npz`
- Teacher trajectory: one `.npz` with distinct forward/reverse IDs, `paired=false`, and both independent ensemble chains
- Figures: all 12 required names per experiment

## Apple Silicon MPS execution

Apple MPS is supported as a hybrid engineering execution backend. Device selection order is CUDA → MPS → CPU. MPS uses `float32/complex64`; CPU/CUDA retain `float64/complex128`. Circuit evolution, differentiable loss values, and parameter gradients run on MPS. Reproducible measurement sampling and POT's detached transport-plan solve are small CPU control operations; detached eigendecomposition diagnostics also run on CPU because MPS lacks complex Hermitian `eigh/eigvalsh`. This backend/precision change is marked △ rather than a paper-method change: circuit/channel/loss definitions are unchanged, but stochastic numerical results need not match CPU exactly.

Engineering smoke artifacts use `smoke_clustered_mps` and `smoke_circular_mps`. Earlier paper-scale MPS runs used the incorrect alternating-CZ circuit and are invalid as reproduction or backend evidence; see [`PAPER_SCALE_MPS_RESULTS.md`](PAPER_SCALE_MPS_RESULTS.md). MPS revalidation is intentionally dropped. CPU is the acceptance backend, while CUDA parity is optional if CUDA deployment is selected.

## Interpretation limits

Smoke runs use smaller `T`, depth, batch, and epochs. They validate code paths, physicality, stochastic reverse blocks, persistence, metrics, and visualization only. Figure 10 uses a separate two-point T sweep explicitly labeled an additional smoke experiment. Cluster outputs include the paper metric `F_data_0` and `F_gen_0`; generic nearest-state fidelity remains diagnostic. Baseline comparison requires clean committed CPU runs with immutable configs and provenance; the paper does not disclose the error-bar protocol or all optimizer settings.
