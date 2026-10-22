from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

from src.corpus_validation.inspection import inspect_html
from src.corpus_validation.paths import local_html_path, normalize_corpus_path, safe_destination

_ARCHIVE: zipfile.ZipFile | None = None
_ROOT: Path | None = None


def initialize_worker(archive: str, root: str) -> None:
    global _ARCHIVE, _ROOT
    _ARCHIVE = zipfile.ZipFile(archive)
    _ROOT = Path(root)


def validate_document(record: dict) -> tuple[dict, dict | None]:
    """Read every byte (including ZIP CRC verification), hash, inspect and materialize."""
    result = {"doc_id": record["doc_id"], "technology": record.get("technology"),
              "category": record.get("category"), "url": record.get("url"), "integrity_ok": False}
    try:
        relative = normalize_corpus_path(record["html_path"])
        raw = _ARCHIVE.read("technical_web_corpus/" + relative)
        digest = hashlib.sha256(raw).hexdigest()
        result.update({"html_bytes": len(raw), "sha256": digest, "html_path": relative})
        if len(raw) != record["html_bytes"]:
            raise ValueError("HTML size disagrees with manifest")
        if digest != record["sha256"]:
            raise ValueError("HTML SHA-256 disagrees with manifest")
        result["integrity_ok"] = True
        local_relative = local_html_path(relative)
        destination = safe_destination(_ROOT, local_relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing extracted file differs; refusing to overwrite it")
        else:
            temporary = destination.with_name(f".raw-{digest[:16]}.{os.getpid()}.part")
            temporary.write_bytes(raw)
            temporary.replace(destination)
        result.update(inspect_html(raw, record))
        normalized = {**record, "html_path": local_relative}
        result["html_path"] = local_relative
        if local_relative != relative:
            normalized.update({"archive_html_path": relative, "source_html_path": record["html_path"],
                               "source_filename": record.get("filename"), "filename": destination.name})
            result["archive_html_path"] = relative
            result["flags"].append("filename_shortened_for_linux")
        return result, normalized
    except Exception as error:
        result.update({"integrity_ok": False, "status": "failed", "reasons": [str(error)], "flags": []})
        return result, None
