"""Textbook scoring, learning and extraction regressions."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from src.neural.scoring import weighted_zone_scores
from src.neural.evaluation.metrics import ndcg,average_precision,classification_metrics
from src.neural.learning.classification.train import _nb_scores,tokens
from src.neural.preprocessing.extract import extract_document
from src.neural.search import SearchEngine


class LearningTests(unittest.TestCase):
    def test_category_routing_respects_explicit_technology_filter(self):
        engine=SearchEngine.__new__(SearchEngine)
        engine.documents=[{'technology':'Python','category':'language'},
                          {'technology':'Docker','category':'containers'}]
        engine.ids=np.array(['python-doc','docker-doc'])
        engine.technology_options=['Python','Docker']
        engine.category_options=['language','containers']
        engine._query=lambda query:np.array([1.,0.])
        engine.classifier=SimpleNamespace(document_classes=np.array([0,1]),
                                          query_scores=lambda *args:np.array([.9,.1]))
        engine.routing={'top_classes':{'centroid':1}}
        engine.weights=np.array([1.,0.,0.,0.])
        allowed_seen=[]
        def zone_scores(vector,allowed):
            allowed_seen.append(allowed.copy())
            return np.array([[.9,-np.inf,-np.inf,-np.inf],
                             [.8,-np.inf,-np.inf,-np.inf]]),np.zeros((2,4),dtype=int),1
        engine.zone_scores=zone_scores
        result=engine.rank('Python',mode='routed-centroid',technology='Docker')
        self.assertEqual(result['selected_classes'],[1])
        self.assertEqual(result['order'].tolist(),[1])
        np.testing.assert_array_equal(allowed_seen[0],[False,True])

    def test_zone_weights_renormalize_missing_zones(self):
        values=np.array([[.2,.8,-np.inf,-np.inf],[-np.inf,-np.inf,.6,-np.inf]])
        score,parts=weighted_zone_scores(values,[.1,.3,.4,.2])
        np.testing.assert_allclose(score,[.65,.6],atol=1e-6)
        np.testing.assert_allclose(parts.sum(axis=1),score,atol=1e-6)

    def test_zone_fallback_and_empty_documents(self):
        score,_=weighted_zone_scores([[-np.inf,.8],[-np.inf,-np.inf]],[1,0])
        self.assertAlmostEqual(score[0],.8,places=6)
        self.assertTrue(np.isneginf(score[1]))

    def test_ndcg_and_map_use_all_judged_relevant_documents(self):
        truth={'a':2,'b':1,'c':0}
        self.assertEqual(ndcg(['a','b','c'],truth),1)
        self.assertAlmostEqual(average_precision(['c','a','b'],truth),7/12)
        self.assertAlmostEqual(average_precision(['c','a'],truth),.25)
        expected=(1+3/math.log2(3))/(3+1/math.log2(3))
        self.assertAlmostEqual(ndcg(['b','a'],truth),expected)

    def test_nb_counts_and_oov_follow_training_vocabulary(self):
        vocabulary={'apple':0,'pear':1}
        likelihood=np.log([[.8,.2],[.2,.8]])
        prior=np.log([.5,.5])
        values=_nb_scores('apple apple pear',vocabulary,likelihood,prior)
        np.testing.assert_allclose(values,np.log([.5*.8*.8*.2,.5*.2*.2*.8]))
        np.testing.assert_allclose(_nb_scores('unknown',vocabulary,likelihood,prior),prior)
        self.assertIn('3.11',tokens('Python 3.11 C++'))
        self.assertIn('c++',tokens('Python 3.11 C++'))

    def test_macro_f1_is_not_accuracy(self):
        result=classification_metrics([0,0,1,1],[0,1,1,1],2)
        self.assertEqual(result['accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],(2/3+4/5)/2)

    def test_table_cells_remain_separate_without_synthetic_headers(self):
        record={'doc_id':'x','title':'API','url':'https://example.com','technology':'test'}
        raw=b'<title>API</title><main><p>This explains the technical fields in the following table.</p><table><tr><th>Name</th><th>Type</th></tr><tr><td>id</td><td>int</td></tr></table></main>'
        result=extract_document(raw,record)
        self.assertIn('Name Type id int',result['body_text'])
        self.assertNotIn('Name: id',result['body_text'])
        self.assertNotIn('table',{b['kind'] for s in result['sections'] for b in s['blocks']})

    def test_standalone_multiline_code_preserves_indentation(self):
        record={'doc_id':'x','title':'API','url':'https://example.com','technology':'test'}
        raw=b'<title>API</title><main><p>This is ordinary text before the source code example in this document.</p><code>if ready:\n    run()</code></main>'
        result=extract_document(raw,record)
        self.assertEqual(result['code_text'],'if ready:\n    run()')
        self.assertNotIn('run()',result['body_text'])
