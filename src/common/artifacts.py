"""Data-stage artifact helpers."""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import sqlite3


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def connect(path, readonly=False):
    if readonly:
        db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.execute('PRAGMA journal_mode=WAL')
    db.row_factory = sqlite3.Row
    return db

def report(directory, title, stats, distributions):
    """Persist compact numeric evidence, charts, and a filterable local dashboard."""
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    (directory / 'charts').mkdir(parents=True, exist_ok=True)
    write_json(directory / 'summary.json', stats)
    figures = []
    for name, values in distributions.items():
        if not values:
            continue
        labels, counts = zip(*values.items())
        fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * .26)))
        ax.barh([str(x) for x in labels], counts, color='#276b80')
        ax.set_xlabel('Count / measured value')
        ax.set_title(name.replace('_', ' ').title())
        fig.tight_layout()
        fig.savefig(directory / 'charts' / f'{name}.png', dpi=125)
        plt.close(fig)
        figures.append(name)
    (directory / 'report.md').write_text('# ' + title + '\n\n```json\n' + json.dumps(stats, indent=2) + '\n```\n\n' + '\n\n'.join(f'![{n}](charts/{n}.png)' for n in figures), encoding='utf-8')
    body = ''.join(f'<figure><img style="max-width:100%" src="charts/{n}.png"><figcaption>{html.escape(n)}</figcaption></figure>' for n in figures)
    rows = ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>' for k,v in stats.items())
    (directory / 'dashboard.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f7fafb;color:#18313a}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ccd}}input{{padding:12px;width:95%}}</style><h1>{html.escape(title)}</h1><label>Filter measurements <input id="filter" type="search"></label><table>{rows}</table>{body}<script>document.querySelector("#filter").oninput=e=>document.querySelectorAll("tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>', encoding='utf-8')

