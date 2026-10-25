"""Package the corrected working corpus for both retrieval engines."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from src.common.artifacts import records, report, digest_file
from src.step_02_validation.paths import safe_destination
from src.step_04_filtering.fix import validate_dataset


def package_corpus(corpus, destination, workers=4):
    corpus, destination = Path(corpus), Path(destination)
    stats = validate_dataset(corpus, workers)
    if not (corpus / "metadata/corpus_state.json").exists():
        raise ValueError("Filter the corpus before publishing its ZIP")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for row in records(corpus / "metadata/corpus_manifest.jsonl"):
            path = safe_destination(corpus, row["html_path"])
            archive.write(path, "technical_web_corpus/" + row["html_path"])
        for name in ("corpus_manifest.jsonl", "corpus_state.json"):
            archive.write(
                corpus / "metadata" / name, "technical_web_corpus/metadata/" + name
            )
    temporary.replace(destination)
    stats.update(
        zip_bytes=destination.stat().st_size, zip_sha256=digest_file(destination)
    )
    report(
        destination.parent,
        "Corrected corpus ZIP",
        stats,
        {"documents": {"retained": stats["documents"]}},
    )
    return stats


def snapshot_downloads(corpus, destination):
    """Preserve a raw collection as the original validation input archive."""
    corpus, destination = Path(corpus), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    manifest = corpus / "metadata/corpus_manifest.jsonl"
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for row in records(manifest):
            archive.write(
                safe_destination(corpus, row["html_path"]),
                "technical_web_corpus/" + row["html_path"],
            )
        archive.write(manifest, "technical_web_corpus/metadata/corpus_manifest.jsonl")
    temporary.replace(destination)
    return destination
