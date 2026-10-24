"""Hand-coded multinomial NB and supervised neural centroid classification."""
from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np

from src.neural.common import digest_file, report, write_json
from src.neural.evaluation.metrics import classification_metrics
from src.neural.learning.zone_weights.zones import ZONES

TOKEN = re.compile(r'c\+\+|c#|[a-z_][a-z0-9_-]*|\d+(?:\.\d+)*',re.I)


def tokens(text):
    return TOKEN.findall(text.casefold())


def normalized(values):
    return values / np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


def build_document_vectors(root, engine):
    """Average all clean window vectors, then normalize each document."""
    sums=np.zeros((len(engine.documents),engine.encoder.dimension),dtype=np.float64)
    counts=np.zeros(len(engine.documents),dtype=np.int64)
    for zone in ZONES:
        vectors,mapping=engine.matrices[zone],engine.mappings[zone]
        for start in range(0,len(vectors),8192):
            block=normalized(np.asarray(vectors[start:start+8192],dtype=np.float32))
            ids=mapping[start:start+len(block)]
            np.add.at(sums,ids,block)
            np.add.at(counts,ids,1)
    if np.any(counts==0):raise ValueError('Some documents have no vectors')
    result=normalized(sums/counts[:,None]).astype(np.float32)
    np.save(Path(root)/'classification/document_vectors.npy',result)
    return result


def _nb_scores(text, vocabulary, likelihood, prior):
    counts=Counter(tokens(text))
    indices,frequencies=[],[]
    for token,count in counts.items():
        if token in vocabulary:
            indices.append(vocabulary[token]);frequencies.append(count)
    return prior+likelihood[:,indices]@np.asarray(frequencies) if indices else prior.copy()


def train_classifiers(root, engine):
    root=Path(root);out=root/'classification';out.mkdir(parents=True,exist_ok=True)
    csv.field_size_limit(16*1024*1024)
    source=Path('data/neural/classification/training.csv')
    with source.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    with Path('data/neural/classification/classes.csv').open() as stream:definitions=list(csv.DictReader(stream))
    classes=sorted(r['class_id'] for r in definitions);class_index={cid:i for i,cid in enumerate(classes)}
    document_index={d['doc_id']:i for i,d in enumerate(engine.documents)}
    targets=np.array([class_index[r['class_id']] for r in rows]);train=np.array([r['split']=='train' for r in rows])
    df=Counter()
    for row in rows:
        if row['split']=='train':df.update(set(tokens(row['text'])))
    vocabulary={token:i for i,token in enumerate(sorted(t for t,n in df.items() if n>=2))}
    counts=np.ones((len(classes),len(vocabulary)),dtype=np.float64)
    documents=np.zeros(len(classes),dtype=np.int64)
    for row,target in zip(rows,targets):
        if row['split']!='train':continue
        documents[target]+=1
        for token,count in Counter(tokens(row['text'])).items():
            if token in vocabulary:counts[target,vocabulary[token]]+=count
    if np.any(documents==0) or not vocabulary:raise ValueError('Each class needs nonempty training examples')
    likelihood=np.log(counts/counts.sum(axis=1,keepdims=True));prior=np.log(documents/documents.sum())
    vectors=build_document_vectors(root,engine)
    selected=np.array([document_index[r['doc_id']] for r in rows])
    centroids=normalized(np.stack([vectors[selected[train & (targets==c)]].mean(axis=0) for c in range(len(classes))])).astype(np.float32)
    with Path('data/neural/classification/technology_classes.csv').open() as stream:technology={r['technology']:r['class_id'] for r in csv.DictReader(stream)}
    all_classes=np.array([class_index[technology[d['technology']]] for d in engine.documents],dtype=np.int16)
    np.savez_compressed(out/'models.npz',log_likelihood=likelihood,log_prior=prior,centroids=centroids,document_classes=all_classes)
    write_json(out/'vocabulary.json',vocabulary)
    results={};predictions=[]
    for method in ('nb','centroid'):
        predicted=[]
        for row,index in zip(rows,selected):
            scores=_nb_scores(row['text'],vocabulary,likelihood,prior) if method=='nb' else centroids@vectors[index]
            predicted.append(int(np.argmax(scores)))
        predicted=np.array(predicted)
        results[method]={}
        for split in ('validation','test'):
            mask=np.array([r['split']==split for r in rows])
            results[method][split]=classification_metrics(targets[mask],predicted[mask],len(classes))
            for row,actual,prediction,use in zip(rows,targets,predicted,mask):
                if use:predictions.append({'method':method,'split':split,'doc_id':row['doc_id'],'actual':classes[actual],'predicted':classes[prediction]})
    with (out/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    model={'index_fingerprint':engine.summary['fingerprint'],'classes':classes,'class_names':{r['class_id']:r['class_name'] for r in definitions},
           'training_csv_sha256':digest_file(source),'vocabulary_size':len(vocabulary),'training_documents':int(train.sum()),
           'nb':'Multinomial token counts, training-only vocabulary with df>=2, add-one smoothing and empirical class priors',
           'centroid':'Mean of normalized training-document vectors; each document averages all pure-zone windows',
           'input_policy':'Original source text only; no added technology/class/URL metadata',
           'label_scope':'Source-domain classification; held-out scores do not validate semantic relevance or short-query routing',
           'results':results}
    write_json(out/'model.json',model)
    report(out/'training','NB and centroid held-out classification',model,
           {'held_out_macro_f1':{f'{method}_{split}':results[method][split]['macro_f1'] for method in results for split in results[method]}})
    print(json.dumps({m:{s:results[m][s]['macro_f1'] for s in results[m]} for m in results}),flush=True)
    return model


class Classifiers:
    def __init__(self, root):
        root=Path(root);directory=root/'classification'
        self.model=json.loads((directory/'model.json').read_text())
        if self.model['index_fingerprint']!=json.loads((root/'zones/summary.json').read_text())['fingerprint']:
            raise RuntimeError('Classifier belongs to an older zone index')
        saved=np.load(directory/'models.npz')
        self.log_likelihood,self.log_prior=saved['log_likelihood'],saved['log_prior']
        self.centroids,self.document_classes=saved['centroids'],saved['document_classes']
        self.vocabulary=json.loads((directory/'vocabulary.json').read_text())

    def query_scores(self, text, vector, method):
        if method=='nb':return _nb_scores(text,self.vocabulary,self.log_likelihood,self.log_prior)
        if method=='centroid':return self.centroids@vector
        raise ValueError('Unknown classifier')

    def probabilities(self, text):
        scores=_nb_scores(text,self.vocabulary,self.log_likelihood,self.log_prior)
        values=np.exp(scores-scores.max())
        return values/values.sum()
