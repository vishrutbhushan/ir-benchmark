"""Recover unusable captures without overwriting original HTML or provenance."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import re
from urllib.parse import urljoin, urlsplit
import requests
from lxml import html
from src.step_02_validation.inspection import inspect_html
from src.common.artifacts import records, report


def _redirect(raw, url):
    tree = html.fromstring(raw)
    refresh = tree.xpath(
        '//meta[translate(@http-equiv,"REFSH","refsh")="refresh"]/@content'
    )
    if refresh:
        m = re.search(r"url\s*=\s*[\'\"]?([^\'\"]+)", refresh[0], re.I)
        if m:
            return urljoin(url, m[1].strip())
    text = raw.decode("utf-8", errors="replace")
    if urlsplit(url).hostname == "kafka.apache.org" and "/documentation/" in url:
        # Current official site replaced these legacy JavaScript redirects.
        version = re.search(r'\b(?:var|let|const)\s+t\s*=\s*[\'"](\d+)[\'"]', text)
        ver = version.group(1) if version else "43"
        path = urlsplit(url).path
        target = (
            path.replace("/documentation/", f"/{ver}/").replace(".html", "")
            if "/streams/" in path
            else f"/{ver}/getting-started/introduction/"
        )
        return urljoin(url, target)
    # Only literal redirects; never execute archived scripts.
    m = re.search(r'(?:window\.)?location\.replace\(\s*[\'"]([^\'"]+)[\'"]\s*\)', text)
    if m:
        return urljoin(url, m[1])
    return None


def repair_one(pair):
    record, check, corpus_root = pair
    trace = {
        "doc_id": record["doc_id"],
        "original_url": record["url"],
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "attempts": [],
    }
    raw = (corpus_root / record["html_path"]).read_bytes()
    target = _redirect(raw, record["url"]) or record["url"]
    session = requests.Session()
    session.headers["User-Agent"] = (
        "IR-Course-Corpus-Repair/1.0 (bounded document recovery)"
    )
    for _ in range(4):
        try:
            response = session.get(target, timeout=(8, 20))
            trace["attempts"].append(
                {
                    "url": target,
                    "status": response.status_code,
                    "final_url": response.url,
                }
            )
            response.raise_for_status()
            raw = response.content
            if len(raw) > 20_000_000:
                raise ValueError("response_too_large")
            updated = {
                **record,
                "url": response.url,
                "original_url": record["url"],
                "content_languages": "",
            }
            evidence = inspect_html(raw, updated)
            if (
                evidence.get("visible_chars", 0) >= 100
                and evidence["status"] != "exclude"
            ):
                sha = hashlib.sha256(raw).hexdigest()
                relative = f"documents/html/repairs/{record['doc_id']}_{sha[:12]}.html"
                destination = corpus_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                updated.update(
                    html_path=relative,
                    sha256=sha,
                    html_bytes=len(raw),
                    source="live_repair",
                    repaired_from_sha256=record["sha256"],
                    fetched_at=trace["attempted_at"],
                )
                trace.update(status="repaired", record=updated, evidence=evidence)
                return trace
            redirect = _redirect(raw, response.url)
            if not redirect or redirect == target:
                trace.update(status="unresolved", evidence=evidence)
                return trace
            target = redirect
        except (requests.RequestException, ValueError) as error:
            trace.update(status="unresolved", error=str(error))
            return trace
    trace["status"] = "unresolved"
    return trace


def run_repairs(corpus_root, validation, out, workers=8):
    out.mkdir(parents=True, exist_ok=True)
    manifests = {
        r["doc_id"]: r for r in records(corpus_root / "metadata/corpus_manifest.jsonl")
    }
    checks = list(records(validation / "document_checks.jsonl"))
    targets = [
        c
        for c in checks
        if set(c.get("reasons", []))
        & {"empty_rendered_content", "javascript_shell", "error_or_challenge_page"}
    ]
    path = out / "repairs.jsonl"
    completed = {r["doc_id"]: r for r in records(path)} if path.exists() else {}
    pending = [c for c in targets if c["doc_id"] not in completed]
    with (
        path.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for result in pool.map(
            repair_one, [(manifests[c["doc_id"]], c, corpus_root) for c in pending]
        ):
            completed[result["doc_id"]] = result
            stream.write(__import__("json").dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"Repairs {len(completed)}/{len(targets)}: {result['doc_id']} {result['status']}",
                flush=True,
            )
    counts = {
        "attempted": len(completed),
        "repaired": sum(r["status"] == "repaired" for r in completed.values()),
    }
    counts["unresolved"] = counts["attempted"] - counts["repaired"]
    report(
        out,
        "Corpus capture repair",
        counts,
        {
            "repair_outcomes": {
                "repaired": counts["repaired"],
                "unresolved": counts["unresolved"],
            }
        },
    )
    return completed
