"""Whole-corpus passage/source correspondence gate, independent of GPU/model loading."""
import json
from itertools import groupby


def audit_passage_coverage(document_db,passage_db):
    groups=iter(groupby(passage_db.execute('SELECT doc_index,section_index,char_start,char_end,tokens,text FROM passages ORDER BY doc_index,section_index,char_start,char_end'),key=lambda row:row['doc_index']))
    documents,sections,characters,passages=0,0,0,0
    for document in document_db.execute('SELECT doc_index,record_json FROM documents ORDER BY doc_index'):
        try:index,group=next(groups)
        except StopIteration:raise RuntimeError('Retained document has no passages')
        if index!=document['doc_index']:raise RuntimeError('Noncontiguous document passage coverage')
        by_section={}
        for row in group:
            if not 0<row['tokens']<=512:raise RuntimeError('Invalid stored token budget')
            by_section.setdefault(row['section_index'],[]).append(row)
            passages+=1
        record=json.loads(document['record_json'])
        if set(by_section)!=set(range(len(record['sections']))):
            raise RuntimeError('Section IDs do not match the source record')
        for section_index,section in enumerate(record['sections']):
            text='\n\n'.join(block['text'] for block in section['blocks'])
            size=len(text);covered=0
            for row in by_section[section_index]:
                start,end=row['char_start'],row['char_end']
                if start>covered or end>size or start<0 or end<=start or row['text']!=text[start:end]:
                    raise RuntimeError(f'Section source/coverage mismatch: {record["doc_id"]}/{section_index}')
                covered=max(covered,end)
            if covered!=size:raise RuntimeError('Source section was truncated or omitted')
            characters+=size;sections+=1
        documents+=1
    try:next(groups)
    except StopIteration:pass
    else:raise RuntimeError('Passages refer to an unknown document')
    return {'documents':documents,'sections':sections,'source_characters':characters,'passages':passages,'all_section_characters_covered':True,'all_passages_match_source_offsets':True}
