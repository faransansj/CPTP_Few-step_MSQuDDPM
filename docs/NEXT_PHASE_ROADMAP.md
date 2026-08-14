# Next Phase Roadmap

This document is planning-only. It does **not** implement a Few-Step model, student, or distillation.

## Current baseline status

- Original 1-qubit MSQuDDPM pipeline, intermediate trajectories, metrics, and 12 figures are implemented.
- The clustered failure was caused by alternating CZ edges instead of every neighboring CZ in every layer.
- Corrected native/TensorCircuit deterministic unitary parity reached maximum error `3.33e-16` against tolerance `1e-12` (`scripts/check_tensorcircuit_parity.py`).
- Corrected clustered CPU diagnostics over seeds 7/42/123 reached `F_gen,0=0.96269±0.02010`, passing the predeclared mean target `0.95`.
- Earlier clustered and circular MPS results used the wrong circuit and are invalidated historical evidence.
- Clean committed CPU reruns, provenance, and baseline tagging remain incomplete.

## Hard gate before future model research

Do not freeze the teacher baseline or start CPTP Few-Step implementation until all are true:

1. Both datasets have at least three physical, reproducible seeds with immutable configs and provenance.
2. Circular remains within a predeclared tolerance of the paper result under a documented metric protocol.
3. Clustered failure is either resolved to a predeclared target (recommended: mean `F_gen,0≥0.95`) **or explicitly bounded** with a signed decision documenting why a nonmatching teacher is acceptable.
4. Every accepted teacher checkpoint has complete step 0–6 trajectories, schema validation, checksums, and environment/commit metadata.
5. A baseline version is tagged and its artifacts are read-only.

Current gate: **Clustered root cause resolved in code; baseline freeze artifacts remain**.

## Backend decision

MPS revalidation is intentionally dropped. The prior MPS failures were produced by the incorrect alternating-CZ implementation and do not demonstrate an Apple Silicon precision defect. CPU is the acceptance backend; CUDA parity remains optional for server deployment. MPS support and configs remain available as non-baseline engineering paths, but no MPS rerun is required for baseline freeze.

## Prioritized TODOs

| Priority | Owner | Work | Artifact | Acceptance check |
|---|---|---|---|---|
| P0 | Reproduction engineer | Finalize corrected circuit and official training-semantics regression tests | code + tests | Full test suite passes; independent review passes |
| P0 | Reproduction engineer | Record immutable provenance | manifests/checksums/config copies | Fresh loader verifies every artifact |
| P0 | Reproduction engineer | Regenerate clustered and circular CPU baselines from a clean commit | checkpoints + trajectories + metrics | Three physical seeds per dataset; clustered mean `F_gen,0≥0.95` |
| P1 | Metrics owner | Match paper circular evaluation protocol, including independent data baseline | metric protocol + CSV | Recomputed result is reproducible from saved states |
| P1 | Validation owner | Run CUDA parity only if CUDA deployment is selected | parity CSV/report | Expected precision differences bounded; physicality passes |
| P1 | Data owner | Freeze teacher trajectory contract | versioned schema + tests | Consumer can select any `rho_t` and rejects incompatible files |
| P2 | Infrastructure owner | Add resumable per-step checkpoints and server job templates | scripts/docs only after review | Interrupted training resumes identically |
| P2 | Visualization owner | Aggregate multi-seed figures/error bars | final figure directory | Figures derive only from manifest-listed runs |

## Completed clustered root-cause analysis

The source-level CZ mismatch is confirmed by TensorCircuit parity and corrected CPU three-seed training. Sampling replay changed seed-42 `F_gen,0` only from `0.94952` to `0.94849`, so it was not the cause of the former `≈0.5` result. The width, ancilla, initialization, optimizer, dataset, and MPS matrices are cancelled to avoid uninformative compute after confirmation of a single root cause.

Every baseline rerun must save: source commit, dirty diff hash, full config, seed, Python/package lock, device/precision, start/end/runtime/RSS, checkpoint hash, history, metrics, forward/reverse/teacher trajectory hashes, physicality report, and parent run ID. Do not promote settings selected from one seed without confirmatory seeds.

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
