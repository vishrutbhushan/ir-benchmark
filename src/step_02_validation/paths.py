import hashlib
from pathlib import Path, PurePosixPath


def normalize_corpus_path(value: str) -> str:
    """Normalize a manifest path without permitting traversal or absolute paths."""
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ":" in path.parts[0] or ".." in path.parts:
        raise ValueError(f"Unsafe corpus path: {value!r}")
    if len(path.parts) < 4 or path.parts[:2] != ("documents", "html"):
        raise ValueError(f"Unexpected HTML path: {value!r}")
    return path.as_posix()


def safe_destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    destination = root / normalize_corpus_path(relative)
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes corpus root: {relative!r}")
    return destination


def local_html_path(relative: str) -> str:
    """ZIP/Windows filenames can exceed Linux's 255-byte component limit."""
    path = PurePosixPath(normalize_corpus_path(relative))
    if any(len(part.encode("utf-8")) > 255 for part in path.parts[:-1]):
        raise ValueError("Directory component exceeds Linux filename byte limit")
    if len(path.name.encode("utf-8")) <= 255:
        return path.as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
    return (path.parent / f"raw_{digest}.html").as_posix()
