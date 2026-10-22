"""Export a shared, source-labeled document/class dataset for NB and Rocchio.

Only data preparation: this neither trains classifiers nor changes retrieval.
"""
from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.neural.common import connect, digest_file, report
from .sample import select_learning_subset


def canonical_family(url):
    parsed = urlsplit(url)
    path = re.sub(r'\d+\.\d+(?:\.\d+)*(?:[-_][\w.-]+)?', 'VERSION', parsed.path)
    path = re.sub(r'(?<=/)(?:v?\d+|stable|latest|current)(?=/|$)', 'VERSION', path)
    return parsed.netloc.lower().removeprefix('www.') + path.rstrip('/')


def export_classification_data(root):
    root = Path(root)
    definitions = Path('data/neural/classification')
    with (definitions / 'classes.csv').open(newline='') as stream:
        classes = {r['class_id']: r for r in csv.DictReader(stream)}
    with (definitions / 'technology_classes.csv').open(newline='') as stream:
        mapping_rows = list(csv.DictReader(stream))
    technology_class = {r['technology']: r['class_id'] for r in mapping_rows}
    if len(technology_class) != len(mapping_rows):
        raise ValueError('Each technology must map to exactly one class')
    if set(technology_class.values()) != set(classes):
        raise ValueError('Every defined class must have mapped technologies')
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    records = []
    with closing(connect(root / 'extraction/corpus.sqlite', True)) as db:
        for row in db.execute('SELECT doc_id,url,title,technology,category,record_json FROM documents ORDER BY doc_id'):
            record = json.loads(row['record_json'])
            content_hash = record['metadata']['content_sha256']
            family = 'url:' + canonical_family(row['url'])
            content = 'text:' + content_hash
            union(family, content)
            cid = technology_class[row['technology']]
            records.append({
                'doc_id': row['doc_id'], 'class_id': cid,
                'class_name': classes[cid]['class_name'], 'technology': row['technology'],
                '_family': family,
            })
    grouped = defaultdict(list)
    for row in records:
        key = find(row.pop('_family'))
        row['group_id'] = hashlib.sha256(key.encode()).hexdigest()[:24]
        grouped[row['group_id']].append(row)
    per_class = defaultdict(list)
    conflicts = 0
    for group_id, members in grouped.items():
        if len({r['class_id'] for r in members}) > 1:
            conflicts += len(members)
            for row in members:
                row.update(split='review', use_for_classification=0,
                           review_reason='shared_content_or_url_family_has_multiple_source_classes')
        else:
            per_class[members[0]['class_id']].append((group_id, members))
    for cid, groups in per_class.items():
        if len(groups) < 3:
            raise ValueError(f'{cid}: fewer than three independent groups; revise class definition')
        n = sum(len(members) for _, members in groups)
        target = {'train': .8*n, 'validation': .1*n, 'test': .1*n}
        assigned = Counter()
        # Largest groups first; stable hash resolves size ties without RNG state.
        groups.sort(key=lambda pair: (-len(pair[1]), hashlib.sha256(('classification-v1:'+pair[0]).encode()).hexdigest()))
        for group_id, members in groups:
            split = max(target, key=lambda name: (target[name]-assigned[name])/target[name])
            assigned[split] += len(members)
            for row in members:
                row.update(split=split, use_for_classification=1, review_reason='')
        if any(assigned[name] == 0 for name in target):
            raise ValueError(f'{cid}: missing split')
    out = root / 'classification'
    out.mkdir(parents=True, exist_ok=True)
    subset_path = definitions / 'training.csv'
    subset = select_learning_subset(records)
    with closing(connect(root / 'extraction/corpus.sqlite', True)) as db:
        with subset_path.open('w', newline='', encoding='utf-8') as stream:
            fields = ['doc_id', 'title', 'text', 'class_id', 'class_name', 'split', 'group_id']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in subset:
                doc = json.loads(db.execute('SELECT record_json FROM documents WHERE doc_id=?', (row['doc_id'],)).fetchone()[0])
                text = '\n\n'.join(filter(None, [doc['title'], '\n'.join(doc['headings']), doc['body_text'], doc['code_text']]))
                writer.writerow({k: row[k] for k in ('doc_id', 'class_id', 'class_name', 'split', 'group_id')} | {'title': doc['title'], 'text': text})
    by_class = Counter(r['class_name'] for r in records)
    stats = {
        'learning_documents': len(subset),
        'learning_splits': dict(Counter(r['split'] for r in subset)),
        'learning_labels_sha256': digest_file(subset_path),
        'documents': len(records), 'classes': len(classes), 'technologies': len(technology_class),
        'eligible_documents': sum(r['use_for_classification'] for r in records),
        'review_documents': conflicts, 'independent_groups': len(grouped),
        'splits': dict(Counter(r['split'] for r in records)),
        'class_counts': dict(by_class),
        'label_status': 'metadata-derived domain labels, not human-reviewed semantic gold',
        'split_policy': 'class-stratified group allocation targeting 80/10/10; canonical version URLs and identical extracted text stay together',
        'limitations': 'Grouping does not catch every near duplicate; held-out results would measure agreement with source labels, not semantic routing quality',
        'text_policy': 'Use only the text column as model input. Class and split columns are labels/control data; all vocabulary, statistics and centroids use training rows only.',
        'classes_sha256': digest_file(definitions / 'classes.csv'),
        'technology_map_sha256': digest_file(definitions / 'technology_classes.csv'),
        'models_trained': False,
    }
    report(out, 'Shared document classification dataset: NB and centroid', stats,
           {'class_documents': dict(by_class), 'split_documents': stats['splits'],
            'learning_class_documents': dict(Counter(r['class_name'] for r in subset)),
            'learning_split_documents': stats['learning_splits']})
    print(json.dumps(stats, indent=2), flush=True)
    print(f'Saved {subset_path}. Both classifiers must use the same split and labels.', flush=True)
    return stats
