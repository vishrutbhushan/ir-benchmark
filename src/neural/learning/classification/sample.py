"""A balanced, reproducible labeled subset shared by both classifiers."""
from collections import defaultdict, deque
import hashlib


SPLIT_SIZES = {'train': 20, 'validation': 5, 'test': 5}


def select_learning_subset(records):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row['use_for_classification']:
            buckets[(row['class_id'], row['split'])][row['technology']].append(row)
    selected = []
    classes = sorted({r['class_id'] for r in records})
    for class_id in classes:
        for split, count in SPLIT_SIZES.items():
            technologies = buckets[(class_id, split)]
            queues = []
            for technology in sorted(technologies):
                rows = sorted(technologies[technology], key=lambda r: hashlib.sha256(r['doc_id'].encode()).hexdigest())
                queues.append(deque(rows))
            used_groups = set()
            chosen = []
            # Round-robin technologies; select only one page from each duplicate group.
            while len(chosen) < count and any(queues):
                for queue in queues:
                    while queue and queue[0]['group_id'] in used_groups:
                        queue.popleft()
                    if queue and len(chosen) < count:
                        row = queue.popleft()
                        chosen.append(row)
                        used_groups.add(row['group_id'])
            if len(chosen) != count:
                raise ValueError(f'{class_id}/{split}: need {count} independent documents')
            selected.extend(chosen)
    return selected
