# Invalidated Historical MPS Results — 1-qubit MSQuDDPM

> **Invalid baseline evidence:** these runs used an incorrect circuit that alternated CZ edges by layer. The official implementation applies every neighboring CZ in every layer. The values below are retained for provenance only and must not be used for reproduction, backend, or precision conclusions. MPS revalidation is intentionally out of scope.

Date: 2026-08-13. Backend: Apple Silicon MPS (`float32/complex64`), with documented CPU control/diagnostic operations. Seeds: 7, 42, 123. The configurations used paper-scale dataset size, `T`, ancilla count, depth, schedule, and Wasserstein loss, but the circuit topology was wrong. LR=0.005, 2001 epochs/step, schedule offset 0.001, and gamma=1 were official-code assumptions rather than fully paper-attested hyperparameters.

## Results

| Dataset | Paper Table I | Our mean ± std (3 seeds) | Status |
|---|---|---|---|
| Clustered | `F_data,0=0.9853±0.0001`, `F_gen,0=0.9873±1e-5` | `F_data,0=0.98588±0.00099`, `F_gen,0=0.50468±0.02203` | Invalidated by wrong CZ topology |
| Circular | `Wass_data=0.0063`, `Wass_gen=0.0151` | generated-vs-target Wasserstein `0.01396±0.00179` | Invalidated by wrong CZ topology |

Additional distribution diagnostics:

| Dataset | Superfidelity | MMD | Runtime/seed |
|---|---:|---:|---:|
| Clustered | `0.64389±0.00948` | `0.56186±0.03766` | `32.10±3.03 min` |
| Circular | `0.99221±0.00333` | `0.00210±0.00216` | `61.07±8.61 min` |

## Per-seed primary values

| Dataset | Seed | Runtime | Primary output |
|---|---:|---:|---|
| Clustered | 7 | 29.36 min | `F_gen,0=0.52070` |
| Clustered | 42 | 35.34 min | `F_gen,0=0.51377` |
| Clustered | 123 | 31.59 min | `F_gen,0=0.47955` |
| Circular | 7 | 56.28 min | Wasserstein `0.01586` |
| Circular | 42 | 55.91 min | Wasserstein `0.01230` |
| Circular | 123 | 71.01 min | Wasserstein `0.01372` |

## Physicality and trajectory contract

All six evaluations exited successfully. Every saved forward/reverse trajectory contains steps 0–6 and passed Hermiticity, trace, PSD, and purity validation. Across all teacher files:

- worst Hermiticity residual: `<4.25e-7`
- worst trace residual: `<2.98e-7`
- global minimum eigenvalue: `>2.7e-6`
- `paired=false`; forward and reverse rows are independent ensemble samples

Seed-42 figures contain all required 12 PNGs for each dataset. Seed-42 quality-vs-step figures use an explicitly additional 4-epoch diagnostic sweep at `T={1,2,4,6}`, not paper-scale trained points.

## Interpretation

These runs show only that the old alternating-CZ implementation produced physical trajectories; they do not measure the official MSQuDDPM circuit or MPS accuracy. Corrected CPU diagnostics are recorded in [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md). No MPS rerun is required for baseline freeze.

Historical raw tables: `outputs/metrics/paper_scale_mps_multiseed.csv`, `outputs/metrics/paper_scale_mps_multiseed_summary.csv`, and `outputs/metrics/paper_scale_mps_multiseed_physicality.json`.
