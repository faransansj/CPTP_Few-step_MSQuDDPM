# Invalidated historical MPS figures

> **Not current reproduction evidence.** These seed-42 images were generated at implementation commit `1c5a16c`, before correcting the reverse-circuit CZ topology. That implementation alternated CZ edges by layer; the official MSQuDDPM circuit applies every neighboring CZ in every layer.

They are retained for provenance and to make the historical failure mode visible rather than hiding it.

- `clustered_seed42/`: pre-fix clustered output, `T=6`, depth 4, 2 ancillas, Wasserstein loss, 2001 epochs/step.
- `circular_seed42/`: pre-fix circular output, `T=6`, depth 8, 2 ancillas, Wasserstein loss, 2001 epochs/step.

Each directory contains 12 historical figures. Figure 10 is additionally only a 4-epoch diagnostic sweep over `T={1,2,4,6}`. See [`../../PAPER_SCALE_MPS_RESULTS.md`](../../PAPER_SCALE_MPS_RESULTS.md) for the invalidation record and [`../../REPRODUCTION_REPORT.md`](../../REPRODUCTION_REPORT.md) for corrected CPU diagnostics.
