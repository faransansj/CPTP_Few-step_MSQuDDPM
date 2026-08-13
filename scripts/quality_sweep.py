#!/usr/bin/env python3
"""Run a bounded additional smoke T sweep; never a Table-I reproduction."""
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from msquddpm.experiment import train_experiment
from msquddpm.utils import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--steps", nargs="+", type=int, default=[1, 2])
p.add_argument("--output-root", default="outputs")
args = p.parse_args()
base = load_config(args.config)
rows = []
for steps in sorted(set(args.steps)):
    config = deepcopy(base)
    config.update(
        {
            "experiment": f"{base['experiment']}_sweep_T{steps}",
            "T": steps,
            "epochs": min(int(base["epochs"]), 4),
            "output_root": args.output_root,
            "paper_scale": False,
        }
    )
    result = train_experiment(config)
    rows.append({"experiment": base["experiment"], "T": steps, **result["metrics"]})
out = Path(args.output_root) / "metrics" / f"{base['experiment']}_quality_vs_steps.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
