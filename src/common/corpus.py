"""Read saved pages from the corpus folder or its ZIP archive."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


class CorpusSource:
    def __init__(self, directory, archive):
        self.directory = Path(directory).resolve()
        self.archive = Path(archive)
        self.prefix = "technical_web_corpus/"
        self.paths = {}
        digest = hashlib.sha256()
        manifest = self.directory / "metadata/corpus_manifest.jsonl"
        if manifest.exists():
            with manifest.open("rb") as stream:
                self._load_manifest(stream, digest)
        else:
            with ZipFile(self.archive) as source:
                with source.open(
                    self.prefix + "metadata/corpus_manifest.jsonl"
                ) as stream:
                    self._load_manifest(stream, digest)
        self.manifest_sha256 = digest.hexdigest()

    def _load_manifest(self, stream, digest):
        for line in stream:
            digest.update(line)
            entry = json.loads(line)
            self.paths[entry["doc_id"]] = entry["html_path"]

    def read_html(self, doc_id):
        raw = self.paths[doc_id]
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
            raise ValueError("Invalid document path")
        path = (self.directory / raw).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Document path escapes the corpus")
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        with ZipFile(self.archive) as source:
            return source.read(self.prefix + raw).decode("utf-8", errors="replace")
