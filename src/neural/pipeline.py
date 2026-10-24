"""Single-runner orchestration of corpus, four-zone indexes and learning."""
import json
from pathlib import Path

from src.neural.common import digest_file, records, write_json


def prepare_corpus(args, root):
    from src.corpus_validation.fix import apply_exclusions, fix_dataset, validate_dataset, REVIEWED_EXCLUSIONS
    validation = args.output_dir/'corpus_validation'
    corrected = (args.corpus_root/'metadata/corpus_state.json').exists()
    if corrected:
        apply_exclusions(args.corpus_root, REVIEWED_EXCLUSIONS, args.output_dir/'corpus_backup/quality')
        validate_dataset(args.corpus_root,args.workers)
        repairs={}
    else:
        from src.corpus_validation.runner import run_validation
        from src.utils.logging_utils import setup_logger
        from src.neural.preprocessing.repair import run_repairs
        saved=validation/'summary.json'
        if not saved.exists() or not json.loads(saved.read_text()).get('integrity_gate_passed'):
            run_validation(args.archive_path,args.corpus_root,validation,setup_logger('corpus_validation',args.logs_dir/'corpus_validation/validation.log'),args.workers)
        repairs=run_repairs(args.corpus_root,validation,root/'repairs',args.workers)
    from src.neural.preprocessing.runner import run_extraction
    run_extraction(args.corpus_root,validation,root/'extraction',repairs,args.workers)
    if not corrected:
        fix_dataset(args.corpus_root,root/'extraction',args.output_dir/'corpus_backup',args.workers)
    else:
        exclusions={r['doc_id']:r['reason'] for r in records(root/'extraction/decisions.jsonl') if r['status']=='excluded'}
        apply_exclusions(args.corpus_root,exclusions,args.output_dir/'corpus_backup/extraction-quality')
    summary=root/'extraction/summary.json'
    saved=json.loads(summary.read_text());saved['dataset_manifest_sha256']=digest_file(args.corpus_root/'metadata/corpus_manifest.jsonl');write_json(summary,saved)
    from src.neural.common import save_configuration
    save_configuration(root)


def run(args):
    root=args.output_dir/'neural';phase=args.phase
    if phase=='audit':
        from src.neural.audit import audit_learning_data
        audit_learning_data(root);return
    if phase=='classification-data':
        from src.neural.learning.classification.dataset import export_classification_data
        export_classification_data(root);return
    if phase in {'prepare','extract','chunk','embed','all','neural-all'}:
        prepare_corpus(args,root)
        if phase in {'prepare','extract'}:return
        from src.neural.zone_index import prepare_zones,embed_zones
        prepare_zones(root,args.workers)
        if phase=='chunk':return
        embed_zones(root,args.device,args.batch_size)
        if phase=='embed':return
    from src.neural.search import SearchEngine
    learning=phase in {'all','neural-all','development-features','weights','classifiers','routing'}
    engine=SearchEngine(root,args.device,args.corpus_root,require_learning=not learning)
    try:
        if phase in {'all','neural-all','development-features','weights'}:
            from src.neural.learning.zone_weights.train import fit_weights,refresh_features
            if phase=='development-features':refresh_features(root,engine);return
            fit_weights(root,engine)
            if phase=='weights':return
        if phase in {'all','neural-all','classifiers'}:
            from src.neural.learning.classification.dataset import export_classification_data
            from src.neural.learning.classification.train import train_classifiers,Classifiers
            export_classification_data(root);train_classifiers(root,engine);engine.classifier=Classifiers(root)
            if phase=='classifiers':return
        if phase in {'all','neural-all','routing'}:
            from src.neural.routing import tune_routing
            tune_routing(root,engine)
            if phase=='routing':return
        if phase=='serve':
            from src.neural.server import serve
            serve(engine,args.host,args.port)
        elif phase=='search':
            if not args.query:raise ValueError('--query is required')
            print(json.dumps(engine.search(args.query,limit=args.limit,technology=args.technology,category=args.category,mode=args.mode),indent=2,ensure_ascii=False))
        elif phase=='evaluate':
            from src.neural.evaluation.runner import evaluate
            evaluate(root,engine,args.classic_rankings)
        elif phase=='evaluation-pool':
            from src.neural.evaluation.pool import make_pool
            make_pool(root,engine)
        else:
            from src.neural.verification import verify
            verify(engine,root,Path('queries.jsonl'))
            if phase in {'all','neural-all'}:
                qrels=Path('data/evaluation/qrels.csv')
                if qrels.exists():
                    from src.neural.evaluation.runner import evaluate
                    evaluate(root,engine,args.classic_rankings)
                print('Complete neural index and learning stages are ready. Run --phase serve.',flush=True)
    finally:
        engine.close()
