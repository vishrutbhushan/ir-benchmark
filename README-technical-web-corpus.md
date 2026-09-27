# Technical Web Corpus Builder

Builds a local corpus of real technical webpages from Common Crawl.

## Install

```bash
python -m pip install -r requirements-technical-web-corpus.txt
```

## Run

```bash
python build_technical_web_corpus.py
```

The script automatically selects the latest Common Crawl crawl. To pin one:

```bash
python build_technical_web_corpus.py --crawl CC-MAIN-2026-34
```

Resume a previous build:

```bash
python build_technical_web_corpus.py --resume
```

## Output

```text
technical_web_corpus/
├── documents/
│   ├── html/                 # original webpages, sharded by hash
│   └── parsed/               # structural representation of each webpage
└── metadata/
    ├── corpus_manifest.jsonl
    ├── corpus_sources.json
    ├── corpus_statistics.json
    └── source_availability.csv
```

The main discovery step is a single bulk query over the Common Crawl URL Index (Parquet). Selected WARC records are then retrieved with HTTP Range requests.

The script keeps a 9 GiB safe corpus budget while treating 10 GiB as the hard output-directory limit.
