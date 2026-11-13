#!/usr/bin/env python3
"""Run classical tokenization and ranked document indexing."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import load_pipeline_config, resolve_path
from src.classical_preprocessing.tokenization_runner import run_tokenization
from src.retrieval.classic_ir_runner import run_classic_ir
from src.utils.logging_utils import setup_logger


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/pipeline.json')
    parser.add_argument('--phase',choices=['2','3','all'],default='all')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    os.environ['IR_CONFIG']=str(args.config.resolve())
    config=load_pipeline_config(args.config)
    settings=config['classical'];paths=config['paths']
    records=resolve_path(paths['classical_records']);queries=resolve_path(paths['queries'])
    outputs=resolve_path(paths['classical_output_dir']);logs=resolve_path(paths['classical_logs_dir'])
    stage2=outputs/'stage_02_tokenization';stage3=outputs/'stage_03_retrieval'
    files=[ROOT/'classical/src/classical_preprocessing'/f for f in ('tokenizer.py','normalizer.py','porter_stemmer.py')]
    token_signature=digest(records)+''.join(digest(p) for p in files)+str(settings['max_docs'])
    stage2_signature=hashlib.sha256(token_signature.encode()).hexdigest()
    if args.phase in {'2','all'}:
        marker=stage2/'stage.done.json'
        complete=marker.exists() and (stage2/'tokenized_corpus.jsonl').exists() and json.loads(marker.read_text()).get('fingerprint')==stage2_signature
        if complete and not args.force: print('Classical tokenization unchanged; reusing it.',flush=True)
        else:
            run_tokenization(records,stage2,setup_logger('tokenization',logs/'tokenization.log'),settings['max_docs'],settings['workers'])
            marker.write_text(json.dumps({'fingerprint':stage2_signature})+'\n')
    if args.phase in {'3','all'}:
        tokenized=stage2/'tokenized_corpus.jsonl'
        source_files=[ROOT/'classical/src/retrieval'/f for f in ('inverted_index.py','bm25_scorer.py','bm25f_scorer.py','query_processor.py')]
        signature=hashlib.sha256((digest(tokenized)+digest(records)+digest(queries)+json.dumps(settings,sort_keys=True)+''.join(digest(p) for p in source_files)).encode()).hexdigest()
        marker=stage3/'stage.done.json'
        complete=marker.exists() and all((stage3/f).exists() for f in ('inverted_index.json','document_catalog.json','bm25_results.json','bm25f_results.json')) and json.loads(marker.read_text()).get('fingerprint')==signature
        if complete and not args.force: print('Classical index and query runs unchanged; reusing them.',flush=True)
        else:
            run_classic_ir(tokenized,records,queries,stage3,setup_logger('retrieval',logs/'retrieval.log'),settings['k1'],settings['b'],settings['top_k'],settings['scorer'],settings['zone_weights'],settings['zone_b'],settings['hierarchical_boost'],settings['routing_mode'])
            marker.write_text(json.dumps({'fingerprint':signature})+'\n')


if __name__=='__main__': main()
