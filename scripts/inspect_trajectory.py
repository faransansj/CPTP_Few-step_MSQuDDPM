#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from msquddpm.states import density_to_bloch, purity
from msquddpm.trajectory import load_trajectory

p=argparse.ArgumentParser(); source=p.add_mutually_exclusive_group(required=True); source.add_argument('--experiment'); source.add_argument('--trajectory'); p.add_argument('--direction',choices=['forward','reverse'],default='reverse'); p.add_argument('--output-root',default='outputs'); args=p.parse_args()
path=Path(args.trajectory) if args.trajectory else Path(args.output_root)/'trajectories'/f'{args.experiment}_{args.direction}.pt'; tr=load_trajectory(path); reports=tr.validate()
summary=[]
for t in tr.steps:
 state=tr.get_state(t); summary.append({'t':t,'shape':list(state.shape),'mean_purity':float(purity(state).mean()),'mean_radius':float(torch.linalg.vector_norm(density_to_bloch(state),dim=1).mean()),'valid':reports[t]['valid']})
print(json.dumps({'path':str(path),'direction':tr.direction,'complete_steps':tr.steps,'states':summary},indent=2))
