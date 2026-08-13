#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
from msquddpm.trajectory import load_trajectory
from msquddpm.visualization import generate_all_figures

p=argparse.ArgumentParser(); p.add_argument('--experiment',required=True); p.add_argument('--output-root',default='outputs'); p.add_argument('--quality-sweep'); args=p.parse_args(); root=Path(args.output_root)
forward=load_trajectory(root/'trajectories'/f'{args.experiment}_forward.pt'); reverse=load_trajectory(root/'trajectories'/f'{args.experiment}_reverse.pt'); history=pd.read_csv(root/'histories'/f'{args.experiment}.csv')
sweep_path=Path(args.quality_sweep) if args.quality_sweep else root/'metrics'/f'{args.experiment}_quality_vs_steps.csv'
sweep=pd.read_csv(sweep_path)
paths=generate_all_figures(forward.get_state(0),forward,reverse,history,root/'figures'/args.experiment,args.experiment,sweep)
print(json.dumps({'figures':[str(x) for x in paths]},indent=2))
