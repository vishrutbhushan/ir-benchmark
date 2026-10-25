"""Four fixed zones over the existing structured document records."""

ZONES = ("title", "headings", "body", "code")


def document_zones(record):
    blocks = [block for section in record["sections"] for block in section["blocks"]]
    return {
        "title": record["title"].strip(),
        "headings": "\n".join(record["headings"]).strip(),
        "body": "\n\n".join(
            b["text"] for b in blocks if b["kind"] in {"prose", "list", "table"}
        ),
        "code": "\n\n".join(b["text"] for b in blocks if b["kind"] == "code"),
    }
