#!/usr/bin/env python3
"""Collection and shared filtering; produces the corpus ZIP consumed by IR branches."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['all','collect','validate','repair','filter','pack'], default='all')
    parser.add_argument('--archive', type=Path, default=Path('original_corpus.zip'))
    parser.add_argument('--corpus', type=Path, default=Path('technical_web_corpus'))
    parser.add_argument('--candidates', type=Path, default=Path('common_crawl_candidates.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    if args.step == 'collect':
        subprocess.run([sys.executable,'src/step_01_collection/collect.py','--candidates',str(args.candidates),'--output',str(args.corpus),'--workers',str(args.workers)],check=True)
        return
    from src.common.artifacts import records, write_json, digest_file
    from src.common.logging import setup_logger
    from src.step_04_filtering.fix import apply_exclusions, fix_dataset, validate_dataset, REVIEWED_EXCLUSIONS
    from src.step_02_validation.runner import run_validation
    from src.step_03_repair.repair import run_repairs
    from src.step_04_filtering.filter import run_extraction
    validation=args.output_dir/'step_02_validation'
    corrected=(args.corpus/'metadata/corpus_state.json').exists()
    if args.step=='pack':
        from src.step_05_archive.pack import package_corpus
        package_corpus(args.corpus,args.output_dir/'step_05_archive/technical_web_corpus.zip',args.workers);return
    if corrected:
        apply_exclusions(args.corpus,REVIEWED_EXCLUSIONS,args.output_dir/'recovery/quality')
        validate_dataset(args.corpus,args.workers);repairs={}
    else:
        run_validation(args.archive,args.corpus,validation,setup_logger('data',Path('logs/data.log')),args.workers)
        if args.step=='validate':return
        repairs=run_repairs(args.corpus,validation,args.output_dir/'step_03_repair',args.workers)
    if args.step in {'validate','repair'}:return
    extraction=args.output_dir/'step_04_filtering'
    run_extraction(args.corpus,validation,extraction,repairs,args.workers)
    if corrected:
        exclusions={r['doc_id']:r['reason'] for r in records(extraction/'decisions.jsonl') if r['status']=='excluded'}
        apply_exclusions(args.corpus,exclusions,args.output_dir/'recovery/content')
    else:
        fix_dataset(args.corpus,extraction,args.output_dir/'recovery',args.workers)
    if args.step=='filter':return
    from src.step_05_archive.pack import package_corpus
    package_corpus(args.corpus,args.output_dir/'step_05_archive/technical_web_corpus.zip',args.workers)

if __name__=='__main__':main()
