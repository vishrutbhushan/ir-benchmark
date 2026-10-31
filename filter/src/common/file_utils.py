from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator, Iterable


def stream_jsonl(file_path: Path | str) -> Generator[dict[str, Any], None, None]:
    """Efficiently streams records from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_num} in {path}: {e}"
                ) from e


def write_jsonl(records: Iterable[dict[str, Any]], file_path: Path | str) -> int:
    """Writes an iterable of records to a JSONL file, creating parent directories."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(data: Any, file_path: Path | str, indent: int = 2) -> None:
    """Writes JSON data with indentation, creating parent directories."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_text(text: str, file_path: Path | str) -> None:
    """Writes plain text / markdown content to disk, creating parent directories."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
