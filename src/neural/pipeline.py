"""The master CLI's independent neural orchestration."""
from pathlib import Path
import json


def run(args):
    root = args.output_dir/'neural'
    (args.logs_dir/'neural').mkdir(parents=True,exist_ok=True)
    phase = args.phase
    if phase == 'classification-data':
        from src.neural.learning.classification.dataset import export_classification_data
        export_classification_data(root)
        return
    if phase == 'development-features':
        from src.neural.learning.zone_weights.features import build_features
        build_features(root, args.device, min(args.batch_size, 32))
        return
    if phase in {'search','serve','verify'}:
        from src.neural.search import SearchEngine
        engine = SearchEngine(root,args.device)
        if phase=='serve':
            from src.neural.server import serve
            serve(engine,args.host,args.port)
        elif phase=='verify':
            from src.neural.verification import verify
            verify(engine,root,Path('queries.jsonl'))
        else:
            if not args.query:
                raise ValueError('--query is required for search')
            print(json.dumps(engine.search(args.query,limit=args.limit,technology=args.technology,category=args.category),indent=2,ensure_ascii=False))
        return
    stages = ['prepare','extract','chunk','embed']
    stop = len(stages)-1 if phase in {'all','neural-all'} else stages.index(phase)
    validation = args.output_dir / 'corpus_validation'
    corrected = (args.corpus_root / 'metadata/corpus_state.json').exists()
    if not corrected:
        summary = validation / 'summary.json'
        valid = json.loads(summary.read_text()) if summary.exists() else {}
        if not valid.get('integrity_gate_passed'):
            from src.corpus_validation.runner import run_validation
            from src.utils.logging_utils import setup_logger
            run_validation(args.archive_path, args.corpus_root, validation,
                           setup_logger('corpus_validation', args.logs_dir / 'corpus_validation/validation.log'), args.workers)
        from src.neural.preprocessing.repair import run_repairs
        repairs = run_repairs(args.corpus_root, validation, root / 'repairs', args.workers)
    else:
        repairs = {}
    from src.neural.preprocessing.runner import run_extraction
    pending_fix = not corrected and (root / 'extraction/eligible_manifest.jsonl').exists() and (root / 'extraction/corpus.sqlite').exists()
    if pending_fix:
        print('Applying the existing complete corpus audit to the dataset itself.', flush=True)
    else:
        run_extraction(args.corpus_root, validation, root / 'extraction', repairs, args.workers)
    if not corrected:
        from src.corpus_validation.fix import fix_dataset
        fix_dataset(args.corpus_root, root / 'extraction', args.output_dir / 'corpus_backup', args.workers)
    if stop == 0:
        return
    from src.neural.common import save_configuration
    save_configuration(root)
    if stop==1:
        return
    from src.neural.model import download_model
    download_model(root/'model')
    from src.neural.preprocessing.chunk import run_chunking
    run_chunking(root/'extraction',root/'passages',root/'model',args.workers)
    from src.neural.preprocessing.integrity import audit_passage_coverage
    from src.neural.common import connect,write_json
    from contextlib import closing
    with closing(connect(root/'extraction/corpus.sqlite',True)) as documents,closing(connect(root/'passages/passages.sqlite',True)) as passages:
        coverage=audit_passage_coverage(documents,passages)
    write_json(root/'passages/coverage.json',coverage)
    print(f'Full source/passage correspondence verified: {coverage}',flush=True)
    if stop==2:
        return
    from src.neural.embedding import run_embeddings
    run_embeddings(root/'passages',root/'embeddings',root/'model',args.device,args.batch_size)
    if phase=='embed':
        return
    from src.neural.search import SearchEngine
    from src.neural.verification import verify
    engine=SearchEngine(root,args.device)
    verify(engine,root,Path('queries.jsonl'))
    print('Complete neural retrieval is ready. Run --phase serve to open local search.',flush=True)
