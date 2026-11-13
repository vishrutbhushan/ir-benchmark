#!/usr/bin/env python3
"""Run the corpus, indexing, retrieval, strict judging and evaluation stages in order."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from config import ROOT,load_pipeline_config,resolve_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/pipeline.json')
    parser.add_argument('--stage',choices=['all','up-to-engine','filter','classical','neural','retrieve','judge','evaluation','search-ui'],default='all')
    args=parser.parse_args()
    config_path=args.config.resolve();config=load_pipeline_config(config_path)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['IR_CONFIG']=env['NEURAL_CONFIG']=str(config_path)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=str(config['execution']['cpu_threads'])
    env['TOKENIZERS_PARALLELISM']='false'
    commands={
        'filter':['filter/run_filter.py','--step','all'],
        'classical':['classical/run_pipeline.py','--phase','all'],
        'neural':['neural/run_pipeline.py','--step','all'],
        'retrieve':['evaluation/run_evaluation.py','--retrieve'],
        'judge':['evaluation/run_evaluation.py','--judge'],
        'evaluation':['evaluation/run_evaluation.py','--metrics','--force'],
        'search-ui':['run_search_ui.py'],
    }
    stages=['filter','classical','neural','retrieve','judge','evaluation'] if args.stage=='all' else ['filter','classical','neural'] if args.stage=='up-to-engine' else [args.stage]
    log_path=resolve_path(config['paths']['pipeline_log']);log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        for stage in stages:
            started=time.perf_counter();print(f'Starting {stage}',flush=True);log.write(f'Starting {stage}\n')
            command=[sys.executable,*commands[stage]]
            if stage!='search-ui':command+=['--config',str(config_path)]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line.rstrip(),flush=True);log.write(line)
            if process.wait():raise SystemExit(f'{stage} failed; completed artifacts remain saved')
            print(f'{stage} completed in {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':main()
