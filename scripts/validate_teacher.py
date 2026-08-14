#!/usr/bin/env python3
"""Strict Teacher schema v1 validation CLI.

Expected config/checkpoint hashes are computed from the given paths, so the
artifact is checked against the exact files on disk, not hand-copied strings.

Usage:
    uv run --locked python scripts/validate_teacher.py TEACHER.npz \
        --config outputs/baselines/clustered_seed7/config.resolved.yaml \
        --checkpoint outputs/baselines/clustered_seed7/checkpoints/clustered_seed7.pt
"""
import argparse
import sys

from msquddpm.provenance import sha256_file
from msquddpm.trajectory import TeacherArtifactError, load_teacher_trajectory

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("teacher", help="Path to the teacher .npz artifact")
parser.add_argument("--dataset", help="Expected dataset name recorded in the artifact metadata")
parser.add_argument("--seed", type=int, help="Expected seed recorded in the artifact metadata")
parser.add_argument("--config", help="Resolved config YAML; its sha256 must match config_sha256")
parser.add_argument("--checkpoint", help="Checkpoint .pt; its sha256 must match checkpoint_sha256")
parser.add_argument("--git-commit", help="Expected source commit recorded in the artifact")
args = parser.parse_args()

try:
    teacher = load_teacher_trajectory(
        args.teacher,
        expected_dataset=args.dataset,
        expected_seed=args.seed,
        expected_config_sha256=sha256_file(args.config) if args.config else None,
        expected_git_commit=args.git_commit,
        expected_checkpoint_sha256=sha256_file(args.checkpoint) if args.checkpoint else None,
    )
except (TeacherArtifactError, OSError) as exc:
    print(f"INVALID: {exc}", file=sys.stderr)
    sys.exit(1)

print(
    f"VALID: {args.teacher} dataset={teacher.metadata['dataset']} seed={teacher.metadata['seed']} "
    f"steps={teacher.forward.steps} physicality_passed={teacher.physicality['passed']}"
)
