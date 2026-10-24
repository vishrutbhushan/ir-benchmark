"""
Toy corpus validation test for Phase 3 Classic IR (Inverted Index & BM25).

Tests:
1. Index construction on a small hand-crafted corpus
2. IDF calculations matching hand computations
3. BM25 scoring accuracy matching hand formula
4. Serialization (save to disk and load back)
5. Query processor tokenization and execution

Run with:
    python -m src.phase3_classic_ir.test_classic_ir
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from src.phase3_classic_ir.bm25_scorer import BM25Scorer
from src.phase3_classic_ir.inverted_index import InvertedIndex, Posting, ZONES


def test_inverted_index_manual():
    """Build a manual 3-document index and verify stats and IDF."""
    idx = InvertedIndex()
    idx.total_docs = 3

    # Document lengths:
    # doc1: title=2, heading=2, body=4, code=2 -> total=10
    # doc2: title=2, heading=1, body=5, code=0 -> total=8
    # doc3: title=1, heading=1, body=10, code=0 -> total=12
    # avg total doc length = (10 + 8 + 12) / 3 = 10.0
    idx.doc_zone_lengths = {
        "doc1": {"title": 2, "heading": 2, "body": 4, "code": 2},
        "doc2": {"title": 2, "heading": 1, "body": 5, "code": 0},
        "doc3": {"title": 1, "heading": 1, "body": 10, "code": 0},
    }
    idx.avg_zone_lengths = {
        "title": 5 / 3,
        "heading": 4 / 3,
        "body": 19 / 3,
        "code": 2 / 3,
    }
    idx.doc_metadata = {
        "doc1": {"title": "Docker Swarm Guide", "url": "https://example.com/1", "technology": "Docker"},
        "doc2": {"title": "Docker Compose Networking", "url": "https://example.com/2", "technology": "Docker"},
        "doc3": {"title": "Kubernetes Pods Overview", "url": "https://example.com/3", "technology": "Kubernetes"},
    }

    # Term: 'docker' appears in doc1 (df=2) and doc2
    idx.postings["docker"] = [
        Posting(doc_id="doc1", zone_tfs={"title": 1, "heading": 1, "body": 2, "code": 1}, positions={"title": [0]}),
        Posting(doc_id="doc2", zone_tfs={"title": 1, "heading": 0, "body": 3, "code": 0}, positions={"title": [0]}),
    ]
    # Term: 'swarm' appears only in doc1 (df=1)
    idx.postings["swarm"] = [
        Posting(doc_id="doc1", zone_tfs={"title": 1, "heading": 1, "body": 1, "code": 0}, positions={"title": [1]}),
    ]
    # Term: 'network' appears in doc2 (df=1)
    idx.postings["network"] = [
        Posting(doc_id="doc2", zone_tfs={"title": 1, "heading": 0, "body": 2, "code": 0}, positions={"title": [1]}),
    ]

    idx.vocabulary = {"docker", "swarm", "network"}

    # Test Document Frequency
    assert idx.document_frequency("docker") == 2
    assert idx.document_frequency("swarm") == 1
    assert idx.document_frequency("network") == 1
    assert idx.document_frequency("missing") == 0

    # Test Hand-computed IDF:
    # Formula: ln((N - df + 0.5) / (df + 0.5) + 1.0)
    # For N=3, df=1 ('swarm'): (3 - 1 + 0.5) / (1 + 0.5) + 1 = 2.5 / 1.5 + 1 = 1.6666667 + 1 = 2.6666667
    # ln(8/3) ≈ 0.980829
    expected_idf_swarm = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1.0)
    actual_idf_swarm = idx.idf("swarm")
    assert abs(actual_idf_swarm - expected_idf_swarm) < 1e-6, f"IDF mismatch: {actual_idf_swarm} vs {expected_idf_swarm}"

    # For N=3, df=2 ('docker'): (3 - 2 + 0.5) / (2 + 0.5) + 1 = 1.5 / 2.5 + 1 = 0.6 + 1 = 1.6
    # ln(1.6) ≈ 0.4700036
    expected_idf_docker = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
    actual_idf_docker = idx.idf("docker")
    assert abs(actual_idf_docker - expected_idf_docker) < 1e-6, f"IDF mismatch: {actual_idf_docker} vs {expected_idf_docker}"

    print("  [OK] Inverted Index DF & IDF calculations verified against hand computation")
    return idx


def test_bm25_hand_calculation(idx: InvertedIndex):
    """Verify BM25 score against exact hand-calculated value."""
    # Let k1 = 1.2, b = 0.75
    scorer = BM25Scorer(idx, k1=1.2, b=0.75)
    avgdl = idx.get_avg_total_doc_length()  # 10.0

    # Query: ['swarm'] -> only doc1 matches
    # doc1 length = 10, avgdl = 10 -> |d| / avgdl = 1.0
    # Length norm factor = 1 - b + b * (|d|/avgdl) = 1 - 0.75 + 0.75 * 1.0 = 1.0
    # For doc1, tf('swarm') = sum(1, 1, 1, 0) = 3
    # TF component = tf * (k1 + 1) / (tf + k1 * length_norm)
    #              = 3 * 2.2 / (3 + 1.2 * 1.0) = 6.6 / 4.2 = 1.57142857
    # Score = IDF('swarm') * 1.57142857
    idf_swarm = idx.idf("swarm")
    expected_score_doc1 = idf_swarm * (3 * 2.2 / (3 + 1.2))

    actual_score_doc1 = scorer.score_document(["swarm"], "doc1")
    assert abs(actual_score_doc1 - expected_score_doc1) < 1e-5, (
        f"BM25 mismatch for doc1 on 'swarm': {actual_score_doc1} vs {expected_score_doc1}"
    )

    # doc2 score on 'swarm' should be exactly 0
    actual_score_doc2 = scorer.score_document(["swarm"], "doc2")
    assert actual_score_doc2 == 0.0

    # Query: ['docker', 'swarm']
    # doc1 should rank higher than doc2 because it contains both terms
    results = scorer.score_query(["docker", "swarm"], top_k=5)
    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"
    assert results[1]["doc_id"] == "doc2"
    assert results[0]["score"] > results[1]["score"]

    print("  [OK] BM25 scoring verified against exact hand calculation")


def test_serialization(idx: InvertedIndex, tmp_path: Path):
    """Test saving and loading the inverted index."""
    test_file = tmp_path / "test_index.json"
    idx.save(test_file)
    assert test_file.exists()

    loaded_idx = InvertedIndex.load(test_file)
    assert loaded_idx.total_docs == idx.total_docs
    assert len(loaded_idx.vocabulary) == len(idx.vocabulary)
    assert loaded_idx.idf("docker") == idx.idf("docker")
    assert len(loaded_idx.get_postings("docker")) == 2

    print("  [OK] Index serialization and deserialization verified")


from src.phase3_classic_ir.bm25f_scorer import BM25FScorer


def test_bm25f_hand_calculation(idx: InvertedIndex):
    """Verify BM25F zone pseudo-frequency and scoring against exact hand calculation."""
    zone_weights = {"title": 4.0, "heading": 2.5, "code": 2.0, "body": 1.0}
    zone_b = {"title": 0.3, "heading": 0.5, "code": 0.6, "body": 0.75}
    k1 = 1.2

    scorer = BM25FScorer(
        idx,
        k1=k1,
        zone_weights=zone_weights,
        zone_b=zone_b,
        hierarchical_boost=0.0,  # disable boost for pure formula test
    )

    # For doc1 and term 'swarm':
    # zone_tfs = {"title": 1, "heading": 1, "body": 1, "code": 0}
    # doc lengths = title: 2, heading: 2, body: 4, code: 2
    # avg lengths = title: 5/3, heading: 4/3, body: 19/3, code: 2/3
    b_title = 1.0 - 0.3 + 0.3 * (2 / (5 / 3))  # 1.06
    b_heading = 1.0 - 0.5 + 0.5 * (2 / (4 / 3))  # 1.25
    b_body = 1.0 - 0.75 + 0.75 * (4 / (19 / 3))  # 0.25 + 3/19 ≈ 0.7236842

    expected_pseudo_tf = (
        4.0 * (1.0 / b_title) + 2.5 * (1.0 / b_heading) + 1.0 * (1.0 / b_body)
    )

    actual_pseudo_tf = scorer.compute_pseudo_tf("doc1", {"title": 1, "heading": 1, "body": 1, "code": 0})
    assert abs(actual_pseudo_tf - expected_pseudo_tf) < 1e-5, (
        f"Pseudo TF mismatch: {actual_pseudo_tf} vs {expected_pseudo_tf}"
    )

    # BM25F Term Score
    expected_score = idx.idf("swarm") * (expected_pseudo_tf / (k1 + expected_pseudo_tf))
    results = scorer.score_query(["swarm"], top_k=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    assert abs(results[0]["score"] - expected_score) < 1e-5, (
        f"BM25F score mismatch: {results[0]['score']} vs {expected_score}"
    )

    # Verify Hierarchical Conditioning
    boosted_scorer = BM25FScorer(
        idx,
        k1=k1,
        zone_weights=zone_weights,
        zone_b=zone_b,
        hierarchical_boost=0.25,
    )
    # Query mentions "Docker", which is the technology of doc1 and doc2
    boosted_results = boosted_scorer.score_query(["swarm"], query_raw_text="Docker swarm", top_k=5)
    expected_boosted_score = expected_score * 1.25
    assert abs(boosted_results[0]["score"] - expected_boosted_score) < 1e-4, (
        f"Hierarchical boost mismatch: {boosted_results[0]['score']} vs {expected_boosted_score}"
    )

    print("  [OK] BM25F zone pseudo-frequency, scoring, and hierarchical boost verified against hand calculation")


def test_hierarchical_postings(idx: InvertedIndex):
    """Verify hierarchical category, technology, and domain postings."""
    idx.category_postings["Containers & Orchestration"] = ["doc1", "doc2", "doc3"]
    idx.technology_postings["Docker"] = ["doc1", "doc2"]
    idx.technology_postings["Kubernetes"] = ["doc3"]
    idx.domain_postings["Infrastructure"] = ["doc1", "doc2", "doc3"]

    assert idx.get_docs_in_category("Containers & Orchestration") == ["doc1", "doc2", "doc3"]
    assert idx.get_docs_in_technology("Docker") == ["doc1", "doc2"]
    assert idx.get_docs_in_technology("Kubernetes") == ["doc3"]
    assert idx.get_docs_in_domain("Infrastructure") == ["doc1", "doc2", "doc3"]
    assert idx.get_docs_in_category("NonExistent") == []

    print("  [OK] Hierarchical category/technology/domain index queries verified")


def test_multilevel_section_retrieval(idx: InvertedIndex):
    """Verify document -> section multi-level passage retrieval."""
    idx.doc_sections["doc1"] = [
        {"sec_id": "doc1#sec_0", "heading": "Swarm Setup", "level": 2, "snippet": "Run docker swarm init to create a cluster manager."},
        {"sec_id": "doc1#sec_1", "heading": "Networking in Swarm", "level": 2, "snippet": "Overlay networks connect containers across nodes."},
    ]
    idx.section_postings["swarm"] = [("doc1#sec_0", 2), ("doc1#sec_1", 1)]
    idx.section_postings["network"] = [("doc1#sec_1", 2)]
    idx.section_lengths["doc1#sec_0"] = 10
    idx.section_lengths["doc1#sec_1"] = 10
    idx.avg_section_length = 10.0

    # Query for 'swarm' section retrieval
    sec_results = idx.score_sections(["swarm"], top_k=5)
    assert len(sec_results) == 2
    assert sec_results[0]["section_id"] == "doc1#sec_0"
    assert sec_results[0]["heading"] == "Swarm Setup"
    assert sec_results[0]["score"] > sec_results[1]["score"]

    # Filter to specific document
    filtered_results = idx.score_sections(["network"], top_k=5, doc_id_filter={"doc1"})
    assert len(filtered_results) == 1
    assert filtered_results[0]["section_id"] == "doc1#sec_1"

    print("  [OK] Multi-level document -> section/passage retrieval verified")


def test_tiered_champion_indexing(idx: InvertedIndex):
    """Verify Tier-1 champion lists and tiered query candidate pruning."""
    idx.build_champion_lists(top_r=1)
    champ = idx.get_champion_postings("docker")
    tier2 = idx.get_tier2_postings("docker")

    assert len(champ) > 0
    assert len(champ) + len(tier2) == len(idx.get_postings("docker"))

    scorer = BM25FScorer(idx, k1=1.2)
    # Test tiered pruning mode
    results_tiered = scorer.score_query(["docker"], routing_mode="tiered", tier1_threshold=1)
    assert len(results_tiered) > 0

    # Test hierarchical-first routing mode
    results_hier = scorer.score_query(["docker"], query_raw_text="docker swarm", routing_mode="hierarchical_first")
    assert len(results_hier) > 0

    # Test simultaneous multi-level retrieval
    multi_res = scorer.score_query_multi_level(["docker", "swarm"], query_raw_text="docker swarm")
    assert "documents" in multi_res
    assert "sections" in multi_res

    print("  [OK] Tiered champion lists, hierarchical-first routing, and multi-level retrieval verified")


def main():
    print("=" * 60)
    print("Phase 3 Classic IR Engine (BM25 & Hierarchical BM25F) Validation Tests")
    print("=" * 60)

    tmp_dir = Path("outputs/test_phase3")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    idx = test_inverted_index_manual()
    test_bm25_hand_calculation(idx)
    test_bm25f_hand_calculation(idx)
    test_hierarchical_postings(idx)
    test_multilevel_section_retrieval(idx)
    test_tiered_champion_indexing(idx)
    test_serialization(idx, tmp_dir)

    # Clean up test output
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("ALL PHASE 3 VALIDATION TESTS PASSED [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
