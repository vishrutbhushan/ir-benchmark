"""Build the document index and export BM25 and BM25F query runs."""

import json
import time
from pathlib import Path
from src.retrieval.inverted_index import InvertedIndex
from src.retrieval.bm25_scorer import BM25Scorer
from src.retrieval.bm25f_scorer import BM25FScorer
from src.retrieval.query_processor import QueryProcessor
from src.utils.file_utils import write_json


def run_classic_ir(tokenized_path, structured_path, queries_path, output_dir, logger,
                   k1, b, top_k, scorer_type, zone_weights, zone_b,
                   hierarchical_boost, routing_mode):
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = InvertedIndex()
    index.build_from_tokenized_corpus(tokenized_path, structured_path, logger)
    from config import load_pipeline_config, resolve_path
    config=load_pipeline_config()
    index.corpus_manifest_sha256=json.loads((resolve_path(config["paths"]["filter_output_dir"])/"stage_04_filtering/summary.json").read_text())["dataset_manifest_sha256"]
    index.save(output_dir / 'inverted_index.json')
    if scorer_type in {'bm25','both'}:
        engine = QueryProcessor(index, BM25Scorer(index,k1=k1,b=b))
        write_json(engine.run_query_suite(queries_path,top_k),output_dir/'bm25_results.json')
    if scorer_type in {'bm25f','both'}:
        scorer = BM25FScorer(index,k1=k1,zone_weights=zone_weights,zone_b=zone_b,hierarchical_boost=hierarchical_boost)
        engine = QueryProcessor(index,scorer,routing_mode=routing_mode)
        write_json(engine.run_query_suite(queries_path,top_k),output_dir/'bm25f_results.json')
    summary = dict(index_statistics=index.get_index_stats(),seconds=time.perf_counter()-started,
                   parameters=dict(k1=k1,b=b,zone_weights=zone_weights,zone_b=zone_b,
                                   hierarchical_boost=hierarchical_boost,routing_mode=routing_mode,top_k=top_k))
    write_json(summary,output_dir/'phase3_summary.json')
    return summary
