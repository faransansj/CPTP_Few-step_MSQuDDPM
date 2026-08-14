# Teacher Artifact Schema v1

Strict, versioned NPZ contract for frozen Original MSQuDDPM teacher baselines.
Implemented in `src/msquddpm/trajectory.py` (`save_teacher_artifact`,
`load_teacher_trajectory`, `TeacherArtifactError`); executable specification in
`tests/test_teacher_artifact.py`.

## Layout

All arrays are stored with `np.savez_compressed`, **no pickle** (`allow_pickle=False`
on load; object arrays are rejected).

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | scalar int | Must equal `1`. |
| `dataset` | scalar string | `clustered` or `circular`. |
| `seed` | scalar int | Run seed (7, 42, or 123 for canonical baselines). |
| `dtype` | scalar string | Must be `complex128`. |
| `validation_tolerance` | scalar float | Physicality tolerance, `1e-8`. |
| `config_sha256` | scalar string | SHA-256 of `config.resolved.yaml`. |
| `git_commit` | scalar string | Source commit. |
| `checkpoint_sha256` | scalar string | SHA-256 of the run checkpoint. |
| `steps` | int array | Must be `[0, 1, 2, 3, 4, 5, 6]`. |
| `rho_0` … `rho_6` | complex128 `(batch, 2, 2)` | Forward chain, actual (batch, d, d) convention. |
| `reverse_rho_0` … `reverse_rho_6` | complex128 `(batch, 2, 2)` | Reverse chain. Both chains are always stored. |
| `forward_sample_id`, `reverse_sample_id` | int arrays | Independent ensemble positions. |
| `sample_id` | int array | Legacy positional index only. |
| `paired` | scalar bool | Always `False`; forward/reverse paths are **not** pairs. |

## Validation (save and load, no repair)

Every density matrix in both chains must be finite, Hermitian, PSD, and unit
trace within `validation_tolerance` (1e-8), checked per sample. Any violation
raises `TeacherArtifactError` with a message identifying the path, key, step,
sample, and reason. Loaders additionally reject: corrupt/unreadable files,
missing keys, wrong `schema_version`, wrong dtype or shape, batch mismatches,
pickled object arrays, and mismatched expected dataset/seed/config/checkpoint
hashes. Values round-trip bit-exactly; nothing is clipped, projected, or
repaired. Successful physicality reports include aggregate and per-step
`passed`, `max_hermitian_error`, `max_trace_error`, `min_eigenvalue`,
`num_failed_states`, and `failed_indices` fields.

## Provenance and policies

- Opted-in runs (`artifact_schema_version: 1` / `provenance_manifest: true`)
  write `config.resolved.yaml`, a strict teacher artifact, and `manifest.json`
  (git commit/dirty identity, environment incl. Python/uv/packages/uv.lock
  hash, timezone-qualified start/end timestamps, runtime, process peak RSS,
  optional parent run ID, metrics, artifact hashes, physicality, success/error).
- **Immutability:** canonical runs claim the output directory atomically and
  fail with `FileExistsError` if it already exists. Teacher NPZ and
  `manifest.json` use temporary-file-plus-replace writes. Failed runs retain an
  atomic manifest with null metrics/physicality and an empty artifact map unless
  the complete validated run succeeds.
- **Canonical preflight** (`provenance.validate_canonical_baseline_config`):
  canonical runs require a clean Git worktree, `device: cpu`,
  `dtype: complex128`, `sampling_semantics: official`, schema v1, and
  tolerance 1e-8. Circular configs whose `circular_acceptance_criterion`
  still starts with `TODO:` are blocked.
- **Clustered acceptance is aggregate-only:** mean `F_gen_0 >= 0.95` across
  seeds 7/42/123; never a per-seed pass/fail.

## Manifest shape

```json
{
  "manifest_schema_version": 1,
  "git": {"commit": "...", "dirty": false, "dirty_identity": null},
  "config": {"path": "config.resolved.yaml", "sha256": "...", "source_path": "configs/baselines/clustered_seed7.yaml", "source_sha256": "..."},
  "environment": {"python_version": "3.11.x", "uv_version": "uv ...", "packages": {}, "platform": "...", "uv_lock_sha256": "..."},
  "device": "cpu",
  "dtype": "complex128",
  "start_time": "2026-08-14T00:00:00+00:00",
  "end_time": "2026-08-14T00:05:00+00:00",
  "runtime_seconds": 300.0,
  "peak_rss_bytes": 123456789,
  "parent_run_id": null,
  "dataset": "clustered",
  "seed": 7,
  "sampling_semantics": "official",
  "artifacts": {"checkpoint_sha256": "...", "trajectory_sha256": "...", "files": {"teacher": {"path": "...", "sha256": "..."}}},
  "physicality": {"passed": true, "steps": {}},
  "metrics": {"F_gen_0": 0.98},
  "success": true,
  "error": null
}
```

## Validating an artifact

```bash
uv run --locked python scripts/validate_teacher.py \
  outputs/baselines/clustered_seed7/trajectories/clustered_seed7_teacher.npz \
  --config outputs/baselines/clustered_seed7/config.resolved.yaml \
  --checkpoint outputs/baselines/clustered_seed7/checkpoints/clustered_seed7.pt
```

Expected hashes are computed from the given paths; the CLI exits non-zero and
prints the failing key/step/sample/reason on any violation.

## Running a canonical baseline

```bash
git status --porcelain  # must be empty
uv run --locked python scripts/train.py --config configs/baselines/clustered_seed7.yaml
```

Circular configs under `configs/baselines/` are intentionally blocked by
preflight while this policy remains unresolved:

> TODO: Finalize Circular baseline evaluation metric, acceptance threshold, and
> tolerance before running the canonical Circular baseline experiments.
