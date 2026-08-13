#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from msquddpm.experiment import load_experiment
from msquddpm.metrics import nearest_metrics
from msquddpm.precision import precision_for
from msquddpm.states import validate_density_matrix

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='auto'); args=p.parse_args()
model,config,target=load_experiment(args.checkpoint,args.device)
mixed=torch.eye(2,dtype=precision_for(target.device).complex,device=target.device)[None].repeat(len(target),1,1)/2
trajectory=model.generate(mixed,return_trajectory=True); validation=trajectory.validate(); metrics=nearest_metrics(trajectory.get_state(0).detach(),target)
out=Path(config.get('output_root','outputs'))/'metrics'/f"{config['experiment']}_evaluation.csv"; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([{'experiment':config['experiment'],'T':config['T'],**metrics}]).to_csv(out,index=False)
print(json.dumps({'metrics':metrics,'trajectory_steps':trajectory.steps,'all_valid':all(x['valid'] for x in validation.values()),'csv':str(out)},indent=2))
