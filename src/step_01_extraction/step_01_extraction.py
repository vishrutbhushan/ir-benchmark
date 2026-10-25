"""Extract prepared corpus documents without collection or filtering policy."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

from src.common.artifacts import connect, digest_file, records, report
from src.common.parallel import bounded_map
from .extract_html import extract_document


def unpack_corpus(archive, corpus):
    corpus, archive = Path(corpus), Path(archive)
    if corpus.exists():
        return
    if not archive.exists():
        raise FileNotFoundError(
            "Place the corrected corpus ZIP from the data branch in data/00_corpus/"
        )
    temporary = corpus.parent / ".unpacking_corpus"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or path.parts[0] != "technical_web_corpus"
            ):
                raise ValueError("Unsafe or unexpected corpus ZIP path")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Corpus ZIP must not contain symbolic links")
        source.extractall(temporary)
    (temporary / "technical_web_corpus").replace(corpus)
    temporary.rmdir()


def _verify(task):
    row, root = task
    path = (Path(root) / row["html_path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Document path escapes the corpus")
    if path.stat().st_size != row["html_bytes"] or digest_file(path) != row["sha256"]:
        raise ValueError(f"Source differs from manifest: {row['doc_id']}")


def _extract(task):
    row, root = task
    return extract_document((Path(root) / row["html_path"]).read_bytes(), row)


def run_extraction(corpus_root, out, workers=4):
    corpus_root, out = Path(corpus_root), Path(out)
    manifest = corpus_root / "metadata/corpus_manifest.jsonl"
    if not (corpus_root / "metadata/corpus_state.json").exists():
        raise ValueError("Use the corrected corpus produced by the data branch")
    inputs = sorted(records(manifest), key=lambda row: row["doc_id"])
    if not inputs or len({row["doc_id"] for row in inputs}) != len(inputs):
        raise ValueError("Corpus document IDs must be nonempty and unique")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_verify, ((row, str(corpus_root)) for row in inputs)))
    listed = {row["html_path"] for row in inputs}
    actual = {
        str(path.relative_to(corpus_root)) for path in corpus_root.rglob("*.html")
    }
    if listed != actual:
        raise ValueError("HTML files and corpus manifest disagree")
    sha, extractor_sha = (
        digest_file(manifest),
        digest_file(Path(__file__).with_name("extract_html.py")),
    )
    summary = out / "summary.json"
    if summary.exists() and all(
        (out / name).exists() for name in ("corpus.sqlite", "structured_records.jsonl")
    ):
        saved = json.loads(summary.read_text())
        if (
            saved["dataset_manifest_sha256"] == sha
            and saved["extractor_sha256"] == extractor_sha
        ):
            print(
                "01: verified prepared corpus; reusing extracted records.", flush=True
            )
            return saved
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "corpus.build.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(str(temporary) + suffix).unlink(missing_ok=True)
    categories, technologies = Counter(), Counter()
    text_path = out / "structured_records.jsonl.part"
    with (
        closing(connect(temporary)) as db,
        text_path.open("w", encoding="utf-8") as stream,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        db.execute(
            "CREATE TABLE documents (doc_index INTEGER PRIMARY KEY,doc_id TEXT UNIQUE NOT NULL,url TEXT,title TEXT,technology TEXT,category TEXT,record_json TEXT)"
        )
        for index, doc in enumerate(
            bounded_map(
                pool, _extract, ((row, str(corpus_root)) for row in inputs), workers * 8
            )
        ):
            row = inputs[index]
            encoded = json.dumps(doc, ensure_ascii=False)
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                (
                    index,
                    doc["doc_id"],
                    doc["url"],
                    doc["title"],
                    row["technology"],
                    row["category"],
                    encoded,
                ),
            )
            stream.write(encoded + "\n")
            categories[row["category"]] += 1
            technologies[row["technology"]] += 1
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    temporary.replace(out / "corpus.sqlite")
    text_path.replace(out / "structured_records.jsonl")
    stats = {
        "fingerprint": hashlib.sha256(
            (sha + extractor_sha + "neural-extraction-v1").encode()
        ).hexdigest(),
        "retained_documents": len(inputs),
        "source_documents": len(inputs),
        "dataset_manifest_sha256": sha,
        "extractor_sha256": extractor_sha,
        "policy": "Prepared data snapshot; no neural-side corpus exclusions",
    }
    report(
        out,
        "01 Neural extraction",
        stats,
        {"categories": dict(categories), "technologies": dict(technologies)},
    )
    return stats
