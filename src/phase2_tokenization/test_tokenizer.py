"""
Quick validation tests for the Phase 2 tokenizer.

Run with:  python -m src.phase2_tokenization.test_tokenizer
"""

from __future__ import annotations

import sys


def test_porter_stemmer():
    """Validate the hand-coded Porter stemmer against known word→stem pairs."""
    from src.phase2_tokenization.porter_stemmer import porter_stem

    # Classic Porter stemmer test pairs
    test_cases = [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("ties", "ti"),
        ("cats", "cat"),
        ("feed", "feed"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("bled", "bled"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("conflated", "conflat"),
        ("troubling", "troubl"),
        ("sized", "size"),
        ("hopping", "hop"),
        ("tanned", "tan"),
        ("falling", "fall"),
        ("hissing", "hiss"),
        ("fizzing", "fizz"),
        ("failing", "fail"),
        ("filing", "file"),
        ("happy", "happi"),
        ("sky", "sky"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("rational", "ration"),
        ("valenci", "valenc"),
        ("hesitanci", "hesit"),
        ("digitizer", "digit"),
        ("conformabli", "conform"),
        ("radicalli", "radic"),
        ("differentli", "differ"),
        ("vileli", "vile"),
        ("analogousli", "analog"),
        ("vietnamization", "vietnam"),
        ("predication", "predic"),
        ("operator", "oper"),
        ("feudalism", "feudal"),
        ("decisiveness", "decis"),
        ("hopefulness", "hope"),
        ("callousness", "callous"),
        ("formaliti", "formal"),
        ("sensitiviti", "sensit"),
        ("sensibiliti", "sensibl"),
        ("triplicate", "triplic"),
        ("formative", "form"),
        ("formalize", "formal"),
        ("electriciti", "electr"),
        ("electrical", "electr"),
        ("hopeful", "hope"),
        ("goodness", "good"),
        ("revival", "reviv"),
        ("allowance", "allow"),
        ("inference", "infer"),
        ("airliner", "airlin"),
        ("gyroscopic", "gyroscop"),
        ("adjustable", "adjust"),
        ("defensible", "defens"),
        ("irritant", "irrit"),
        ("replacement", "replac"),
        ("adjustment", "adjust"),
        ("dependent", "depend"),
        ("adoption", "adopt"),
        ("homologou", "homolog"),
        ("communism", "commun"),
        ("activate", "activ"),
        ("angulariti", "angular"),
        ("homologous", "homolog"),
        ("effective", "effect"),
        ("bowdlerize", "bowdler"),
        ("probate", "probat"),
        ("rate", "rate"),
        ("cease", "ceas"),
        ("controll", "control"),
        ("roll", "roll"),
    ]

    passed = 0
    failed = 0
    for word, expected in test_cases:
        result = porter_stem(word)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: porter_stem('{word}') = '{result}', expected '{expected}'")
            failed += 1

    print(f"Porter Stemmer: {passed}/{passed + failed} tests passed")
    return failed == 0


def test_version_strings():
    """Verify version strings are detected and kept atomic."""
    from src.phase2_tokenization.normalizer import is_version_string, expand_technical_token

    # Should be detected as version strings
    versions = ["3.11", "v2.4", "1.0.0", "v10.15.3", "ES2015", "JDK21", "v2", "2.0-beta"]
    for v in versions:
        assert is_version_string(v), f"Expected '{v}' to be a version string"
        expanded = expand_technical_token(v)
        assert len(expanded) == 1, f"Version '{v}' should not be split, got: {expanded}"
        print(f"  OK: '{v}' -> {expanded}")

    # Should NOT be detected as version strings
    non_versions = ["python", "react", "hello-world", "get_data"]
    for nv in non_versions:
        assert not is_version_string(nv), f"'{nv}' should NOT be a version string"

    print("Version string tests: all passed")
    return True


def test_technical_identifiers():
    """Verify technical identifier expansion produces split + unsplit forms."""
    from src.phase2_tokenization.normalizer import expand_technical_token

    test_cases = [
        ("react-router-dom", ["react-router-dom", "react", "router", "dom"]),
        ("find_unused_parameters", ["find_unused_parameters", "find", "unused", "parameters"]),
        ("python", ["python"]),  # No split
    ]

    for token, expected in test_cases:
        result = expand_technical_token(token)
        assert result == expected, f"expand('{token}') = {result}, expected {expected}"
        print(f"  OK: '{token}' -> {result}")

    # CamelCase
    result = expand_technical_token("getElementById")
    assert "getelementbyid" in result, f"Missing full form: {result}"
    assert "get" in result, f"Missing 'get': {result}"
    assert "element" in result, f"Missing 'element': {result}"
    print(f"  OK: 'getElementById' -> {result}")

    print("Technical identifier tests: all passed")
    return True


def test_tokenizer_pipeline():
    """Test the full tokenization pipeline on sample technical text."""
    from src.phase2_tokenization.tokenizer import tokenize

    # Test 1: Basic technical text
    text = "How to configure react-router-dom v2.4 for single page applications"
    tokens = tokenize(text)
    print(f"  Input: '{text}'")
    print(f"  Tokens: {tokens}")

    # Version should be preserved atomic
    assert "v2.4" in tokens, f"Version 'v2.4' should be in tokens: {tokens}"
    # Identifier should have both split and unsplit forms (after stemming)
    assert "react-router-dom" in tokens or "react" in tokens, f"Missing identifier forms: {tokens}"

    # Test 2: Code-like text
    code_text = "import React from 'react'; const App = () => { return <div>Hello</div>; }"
    code_tokens = tokenize(code_text, stem=False, remove_stopwords=False)
    print(f"  Code tokens: {code_tokens}")

    # Test 3: Text with version and identifiers
    text3 = "Python 3.11 asyncio.run() and find_unused_parameters=True"
    tokens3 = tokenize(text3)
    print(f"  Input: '{text3}'")
    print(f"  Tokens: {tokens3}")
    assert "3.11" in tokens3, f"Version '3.11' should be atomic: {tokens3}"

    print("Full pipeline tests: all passed")
    return True


def test_zone_tokenization():
    """Test per-zone tokenization on a sample document record."""
    from src.phase2_tokenization.tokenizer import tokenize_zones

    record = {
        "doc_id": "test_001",
        "title": "Docker multi-stage build caching syntax",
        "headings": [
            {"level": 1, "text": "Docker Build Optimization"},
            {"level": 2, "text": "Using --mount=type=cache"},
        ],
        "sections": [
            {"heading": "Introduction", "level": 1, "content": "Docker supports multi-stage builds for efficient image creation."},
        ],
        "body_text": "Docker multi-stage builds allow you to use multiple FROM statements in your Dockerfile. Version 3.11 of Python is recommended.",
        "code_text": "FROM python:3.11-slim\nRUN pip install fastapi\nCOPY --from=builder /app /app",
        "metadata": {"technology": "Docker", "category": "containers-orchestration"},
    }

    result = tokenize_zones(record)
    print(f"  doc_id: {result['doc_id']}")
    print(f"  title_tokens ({len(result['title_tokens'])}): {result['title_tokens']}")
    print(f"  heading_tokens ({len(result['heading_tokens'])}): {result['heading_tokens']}")
    print(f"  body_tokens ({len(result['body_tokens'])}): {result['body_tokens'][:20]}...")
    print(f"  code_tokens ({len(result['code_tokens'])}): {result['code_tokens'][:15]}...")
    print(f"  section_tokens ({len(result['section_tokens'])}): {result['section_tokens']}")
    print(f"  total_token_count: {result['total_token_count']}")

    assert result["doc_id"] == "test_001"
    assert len(result["title_tokens"]) > 0
    assert result["total_token_count"] > 0

    print("Zone tokenization tests: all passed")
    return True


def main():
    print("=" * 60)
    print("Phase 2 Tokenizer Validation Tests")
    print("=" * 60)

    all_passed = True

    print("\n--- Porter Stemmer ---")
    all_passed &= test_porter_stemmer()

    print("\n--- Version Strings ---")
    all_passed &= test_version_strings()

    print("\n--- Technical Identifiers ---")
    all_passed &= test_technical_identifiers()

    print("\n--- Full Pipeline ---")
    all_passed &= test_tokenizer_pipeline()

    print("\n--- Zone Tokenization ---")
    all_passed &= test_zone_tokenization()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
        sys.exit(1)


if __name__ == "__main__":
    main()
