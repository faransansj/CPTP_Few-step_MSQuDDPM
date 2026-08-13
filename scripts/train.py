#!/usr/bin/env python3
import argparse, json
from msquddpm.experiment import train_experiment
from msquddpm.utils import load_config

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
r=train_experiment(load_config(args.config))
print(json.dumps({'checkpoint':str(r['checkpoint']),'metrics':r['metrics'],'history_rows':len(r['history'])},indent=2))
