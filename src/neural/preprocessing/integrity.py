"""Whole-corpus fixed-zone coverage and vector/document correspondence."""
import json
from itertools import groupby

from src.neural.learning.zone_weights.zones import ZONES,document_zones


def audit_zone_coverage(document_db, stores):
    groups={z:iter(groupby(stores[z].execute('SELECT doc_index,char_start,char_end,tokens,text FROM passages ORDER BY doc_index,passage_index'),key=lambda row:row['doc_index'])) for z in ZONES}
    pending={z:next(groups[z],None) for z in ZONES}
    characters={z:0 for z in ZONES};windows={z:0 for z in ZONES};documents=0
    for document in document_db.execute('SELECT doc_index,record_json FROM documents ORDER BY doc_index'):
        source=document_zones(json.loads(document['record_json']))
        for zone in ZONES:
            text=source[zone];size=len(text);covered=0
            current=pending[zone]
            if current is not None and current[0]<document['doc_index']:raise RuntimeError('Unknown/out-of-order document in zone store')
            if current is not None and current[0]==document['doc_index']:
                for row in current[1]:
                    start,end=row['char_start'],row['char_end']
                    if not 0<row['tokens']<=512 or start<0 or start>covered or end<=start or end>size or row['text']!=text[start:end]:
                        raise RuntimeError(f'Zone source/coverage mismatch: {document["doc_index"]}/{zone}')
                    covered=max(covered,end);windows[zone]+=1
                pending[zone]=next(groups[zone],None)
            if covered!=size:raise RuntimeError(f'Zone source text truncated or omitted: {document["doc_index"]}/{zone}')
            characters[zone]+=size
        documents+=1
    if any(value is not None for value in pending.values()):raise RuntimeError('Zone references an unknown document')
    return {'documents':documents,'characters':characters,'windows':windows,'all_zone_characters_covered':True}


def audit_vector_mapping(passage_db, doc_indices):
    count=0
    for row in passage_db.execute('SELECT passage_index,doc_index FROM passages ORDER BY passage_index'):
        if row['passage_index']!=count or count>=len(doc_indices) or doc_indices[count]!=row['doc_index']:
            raise RuntimeError('Vector/document mapping differs from passage metadata')
        count+=1
    if count!=len(doc_indices):raise RuntimeError('Vector/document mapping length differs from passages')
    return count
