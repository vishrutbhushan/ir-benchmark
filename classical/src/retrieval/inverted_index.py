"""Build, persist and load the four-zone positional inverted index."""

import json
import sys
from array import array
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple
from src.utils.taxonomy import get_taxonomy_for_tech

ZONES = ("title", "heading", "body", "code")

def iter_terms(path):
    """Stream every term without duplicating the complete JSON index in memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, position, ended = "", 0, False

        def refill():
            nonlocal buffer, position, ended
            chunk = stream.read(1024 * 1024)
            buffer = buffer[position:] + chunk
            position = 0
            ended = not chunk

        def skip_space():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                refill()

        def consume(character):
            nonlocal position
            skip_space()
            if position >= len(buffer) or buffer[position] != character:
                raise ValueError(f"Expected JSON separator {character!r}")
            position += 1

        def read_value():
            nonlocal position
            skip_space()
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    return value
                except json.JSONDecodeError:
                    if ended:
                        raise
                    refill()

        consume("{")
        while True:
            skip_space()
            if position < len(buffer) and buffer[position] == "}":
                break
            key = read_value()
            consume(":")
            value = read_value()
            yield key, value
            skip_space()
            if position < len(buffer) and buffer[position] == "}":
                break
            consume(",")


class Posting(NamedTuple):
    doc_id: str
    zone_tfs: dict
    positions: dict


class InvertedIndex:
    def __init__(self):
        self.postings = {}
        self.doc_zone_lengths = {}
        self.avg_zone_lengths = {}
        self.total_docs = 0
        self.doc_metadata = {}
        self.vocabulary = set()
        self.category_postings = defaultdict(list)
        self.technology_postings = defaultdict(list)
        self.domain_postings = defaultdict(list)
        self.corpus_manifest_sha256 = None

    def build_from_tokenized_corpus(self, tokenized_path, structured_path, logger):
        started = time.perf_counter()
        with Path(structured_path).open(encoding="utf-8") as source:
            for line in source:
                doc = json.loads(line)
                tech = doc["metadata"]["technology"]
                taxonomy = get_taxonomy_for_tech(tech)
                category = taxonomy["category"] if taxonomy["category"] != "Other" else doc["metadata"]["category"]
                self.doc_metadata[doc["doc_id"]] = dict(title=doc["title"], url=doc["url"], technology=tech, category=category, domain=taxonomy["domain"])
                self.category_postings[category].append(doc["doc_id"])
                self.technology_postings[tech].append(doc["doc_id"])
                self.domain_postings[taxonomy["domain"]].append(doc["doc_id"])
        totals = dict.fromkeys(ZONES, 0)
        with Path(tokenized_path).open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                doc = json.loads(line)
                did = doc["doc_id"]
                local = {}
                lengths = {}
                for zone in ZONES:
                    tokens = doc[zone + "_tokens"]
                    lengths[zone] = len(tokens)
                    totals[zone] += len(tokens)
                    for position, token in enumerate(tokens):
                        token = token.replace("\ufffd", "").strip()
                        if not token or len(token) > 64:
                            continue
                        local.setdefault(token, {}).setdefault(zone, []).append(position)
                self.doc_zone_lengths[did] = lengths
                for term, positions in local.items():
                    self.postings.setdefault(term, []).append(Posting(did, {z:len(v) for z,v in positions.items()}, {z:array("I",v) for z,v in positions.items()}))
                if number % 5000 == 0:
                    logger.info("Indexed %s documents, %s terms", number, len(self.postings))
        self.total_docs = len(self.doc_zone_lengths)
        if set(self.doc_zone_lengths) != set(self.doc_metadata):
            raise ValueError("Tokenized documents and source metadata disagree")
        self.avg_zone_lengths = {z:totals[z]/self.total_docs for z in ZONES}
        self.vocabulary = set(self.postings)
        logger.info("Built complete index in %.1f seconds", time.perf_counter()-started)

    def get_postings(self, term):
        return self.postings.get(term, [])

    def document_frequency(self, term):
        return len(self.get_postings(term))

    def idf(self, term):
        df = self.document_frequency(term)
        return math.log((self.total_docs-df+0.5)/(df+0.5)+1) if df else 0.0

    def get_zone_length(self, doc_id, zone):
        return self.doc_zone_lengths.get(doc_id, {}).get(zone, 0)

    def get_total_doc_length(self, doc_id):
        return sum(self.doc_zone_lengths.get(doc_id, {}).values())

    def get_avg_total_doc_length(self):
        return sum(self.avg_zone_lengths.values())

    def get_docs_in_category(self, category):
        return self.category_postings.get(category, [])

    def get_index_stats(self):
        return dict(total_documents=self.total_docs, vocabulary_size=len(self.vocabulary),
                    total_postings_entries=sum(map(len,self.postings.values())),
                    avg_zone_lengths=self.avg_zone_lengths)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.part")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write("{")
            for i,(term,postings) in enumerate(self.postings.items()):
                if i: stream.write(",")
                stream.write(json.dumps(term,ensure_ascii=False)+":")
                json.dump({"postings":[dict(doc_id=p.doc_id,tfs=p.zone_tfs,positions={z:list(v) for z,v in p.positions.items()}) for p in postings]},stream,ensure_ascii=False,separators=(",",":"))
            stream.write("}")
        temporary.replace(path)
        catalog = {key:getattr(self,key) for key in ("total_docs","avg_zone_lengths","doc_zone_lengths","doc_metadata","category_postings","technology_postings","domain_postings","corpus_manifest_sha256")}
        temporary = path.parent/"document_catalog.json.part"
        temporary.write_text(json.dumps(catalog,ensure_ascii=False),encoding="utf-8")
        temporary.replace(path.parent/"document_catalog.json")

    @classmethod
    def load(cls, path, catalog_path=None):
        path = Path(path)
        catalog_path = Path(catalog_path) if catalog_path else path.parent/"document_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        index = cls()
        for key,value in catalog.items(): setattr(index,key,value)
        document_ids={did:did for did in index.doc_metadata}
        for term,value in iter_terms(path):
            index.postings[term] = [Posting(document_ids[p["doc_id"]],{sys.intern(z):v for z,v in p["tfs"].items()},{sys.intern(z):array("I",v) for z,v in p["positions"].items()}) for p in value["postings"]]
        index.vocabulary = set(index.postings)
        return index
