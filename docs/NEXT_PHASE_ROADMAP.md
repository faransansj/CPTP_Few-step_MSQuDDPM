# Next Phase Roadmap

This document is planning-only. It does **not** implement a Few-Step model, student, or distillation.

## Current baseline status

- Original 1-qubit MSQuDDPM pipeline, MPS execution, intermediate trajectories, metrics, and 12 figures are implemented.
- Paper-scale MPS runs completed for clustered and circular datasets with seeds 7, 42, 123; all six trajectories are physical and complete through `T=6`.
- Circular is close to Table I: Wasserstein `0.01396±0.00179` versus paper generated `0.0151`.
- Clustered failed consistently: `F_gen,0=0.50468±0.02203` versus paper `0.9873`; target data metric matches (`0.98588±0.00099` versus `0.9853`).
- Evidence: [`PAPER_SCALE_MPS_RESULTS.md`](PAPER_SCALE_MPS_RESULTS.md), `outputs/metrics/paper_scale_mps_multiseed*.{csv,json}`, checkpoints, logs, and teacher trajectories.

## Hard gate before future model research

Do not freeze the teacher baseline or start CPTP Few-Step implementation until all are true:

1. Both datasets have at least three physical, reproducible seeds with immutable configs and provenance.
2. Circular remains within a predeclared tolerance of the paper result under a documented metric protocol.
3. Clustered failure is either resolved to a predeclared target (recommended: mean `F_gen,0≥0.95`) **or explicitly bounded** with a signed decision documenting why a nonmatching teacher is acceptable.
4. Every accepted teacher checkpoint has complete step 0–6 trajectories, schema validation, checksums, and environment/commit metadata.
5. A baseline version is tagged and its artifacts are read-only.

Current gate: **BLOCKED by clustered reproduction**.

## Prioritized TODOs

| Priority | Owner | Work | Artifact | Acceptance check |
|---|---|---|---|---|
| P0 | Reproduction engineer | Run clustered root-cause matrix below | `outputs/ablations/clustered/*.csv` | At least 3 seeds for promoted setting; physical trajectories; result explained |
| P0 | Quantum reviewer | Establish official TensorCircuit parity on one deterministic batch/block | parity notebook/report + tensors | Unitary/output/loss agree within precision tolerance |
| P0 | Research lead | Decide resolved vs explicitly bounded clustered baseline | signed decision in reproduction report | Hard gate outcome unambiguous |
| P0 | Reproduction engineer | Record immutable provenance | manifests/checksums/config copies | Fresh loader verifies every artifact |
| P1 | Metrics owner | Match paper circular evaluation protocol, including independent data baseline | metric protocol + CSV | Recomputed result is reproducible from saved states |
| P1 | Validation owner | CPU/CUDA cross-backend parity on selected seeds | parity CSV/report | Expected precision differences bounded; physicality passes |
| P1 | Data owner | Freeze teacher trajectory contract | versioned schema + tests | Consumer can select any `rho_t` and rejects incompatible files |
| P2 | Infrastructure owner | Add resumable per-step checkpoints and server job templates | scripts/docs only after review | Interrupted training resumes identically |
| P2 | Visualization owner | Aggregate multi-seed figures/error bars | final figure directory | Figures derive only from manifest-listed runs |

## Clustered root-cause experiment matrix

Change one factor at a time from the current paper-v2/MPS seed-42 baseline, then promote plausible settings to seeds 7/42/123.

| Factor | Levels | Why |
|---|---|---|
| Cluster width | paper v2 `epsilon=0.08`; official code `0.04` | Concrete paper/code discrepancy |
| Ancilla | Haar+zero; all-zero | Paper reports both; isolates stochastic input effect |
| Initialization | normal; Xavier with normal ancilla parameters | Both described by paper/official code |
| Optimizer | LR `{1e-3, 5e-3, 1e-2}`; decay `{gamma=1, documented exponential candidates}`; epochs `{500,2001,4000}` | Paper omits exact per-task schedule |
| Sampling semantics | fixed Haar per stage + fresh measurement; fully fresh; official-code-exact | Current interpretation may differ from artifact-generating code |
| Framework | native PyTorch; official TensorCircuit | Detect simulator/gate/measurement parity errors |
| Dataset semantics | paper-normalized recipe; official generator exact | Rule out sample construction differences |
| Precision/backend | MPS complex64; CPU/CUDA complex128 | Bound low-precision impact |

Every run must save: source commit, dirty diff hash, full config, seed, Python/package lock, device/precision, start/end/runtime/RSS, checkpoint hash, history, metrics, forward/reverse/teacher trajectory hashes, physicality report, and parent run ID. Do not promote settings selected from one seed without confirmatory seeds.

## Baseline freeze and teacher trajectory contract

When the gate passes:

1. Tag baseline (proposed `msquddpm-teacher-v1`) and record commit/config/checkpoint SHA-256 hashes.
2. Freeze one manifest per dataset/seed; never overwrite artifacts.
3. Teacher NPZ schema must include `schema_version`, `forward_sample_id`, `reverse_sample_id`, `paired=false`, `steps`, `rho_0...rho_T`, `reverse_rho_0...reverse_rho_T`, dataset/config/seed/commit/checkpoint hashes, dtype, and physicality tolerance.
4. Forward/reverse indices remain independent ensemble positions; no false paired-path claim.
5. Loader tests must retrieve `get_state(t)`, reject missing/wrong shapes, verify hashes, and validate every density matrix.

## Future CPTP Few-Step research — only after the gate

No code is authorized by this roadmap. Once the teacher is frozen, treat this as a separate research project:

1. Define student channel interfaces for `8→4` and `4→0` (or teacher-supported corresponding schedule).
2. Specify CPTP parameterization and proofs/tests: Hermiticity preservation, trace preservation, Choi PSD, complete positivity with ancilla extension, batch physicality, and numerical tolerance.
3. Define distribution/trajectory distillation targets without assuming forward/reverse row pairing.
4. Train and validate `rho_8→rho_4→rho_0` against held-out teacher ensembles.
5. Compare quality/runtime/physicality to the frozen Original MSQuDDPM teacher.

## Compute estimates and parallel CUDA plan

Observed Apple MPS paper-scale runtime per seed:

- clustered: ~29–35 min (mean 32 min)
- circular: ~56–71 min (mean 61 min)
- both datasets × 3 seeds: ~4.7 serial hours observed

Root-cause screening matrix can grow to dozens of runs. On CUDA servers, parallelize **independent config/seed jobs only**, one process/GPU initially. Safe parallel lanes:

- epsilon/ancilla/init screens on separate GPUs
- optimizer schedules on separate GPUs
- seeds only after a setting is promoted
- TensorCircuit parity and metric recomputation independently from training

Do not parallelize writes to the same output directory or reuse experiment names. Each job gets a unique immutable artifact root. Benchmark one job per GPU first; estimate memory and runtime rather than assuming CUDA speedup. Aggregate only after all manifests and exit codes validate.
