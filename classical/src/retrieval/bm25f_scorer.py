"""BM25F over four zones, with configured taxonomy routing and boosting."""

from collections import defaultdict
from config import load_pipeline_config
from src.retrieval.inverted_index import ZONES
from src.utils.taxonomy import infer_query_taxonomy


class BM25FScorer:
    def __init__(self,index,k1=None,zone_weights=None,zone_b=None,hierarchical_boost=None):
        settings=load_pipeline_config()['classical']
        self.index=index
        self.k1=settings['k1'] if k1 is None else k1
        self.zone_weights=dict(settings['zone_weights'] if zone_weights is None else zone_weights)
        self.zone_b=dict(settings['zone_b'] if zone_b is None else zone_b)
        self.hierarchical_boost=settings['hierarchical_boost'] if hierarchical_boost is None else hierarchical_boost
        if set(self.zone_weights)!=set(ZONES) or set(self.zone_b)!=set(ZONES):
            raise ValueError('BM25F must configure all four zones')

    def compute_pseudo_tf(self,doc_id,zone_tfs):
        value=0.0
        for zone,tf in zone_tfs.items():
            average=self.index.avg_zone_lengths[zone]
            b=self.zone_b[zone]
            normalization=(1-b)+b*self.index.get_zone_length(doc_id,zone)/average if average else 1.0
            value+=self.zone_weights[zone]*tf/normalization
        return value

    def score_query(self,query_terms,query_raw_text='',top_k=20,category_filter=None,routing_mode='hybrid'):
        taxonomy=infer_query_taxonomy(query_raw_text) if query_raw_text else {}
        technologies=set(taxonomy.get('matched_technologies',[]))
        categories=set(taxonomy.get('matched_categories',[]))
        if routing_mode=='hierarchical_first' and category_filter is None and categories:
            selected=self.score_query(query_terms,query_raw_text,top_k,sorted(categories)[0],'hybrid')
            if len(selected)>=top_k:return selected
        elif routing_mode not in ('hybrid','hierarchical_first'):
            raise ValueError('Unsupported classical routing mode')
        allowed=set(self.index.get_docs_in_category(category_filter)) if category_filter else None
        scores=defaultdict(float)
        for term in query_terms:
            idf=self.index.idf(term)
            for posting in self.index.get_postings(term):
                if allowed is not None and posting.doc_id not in allowed:continue
                tf=self.compute_pseudo_tf(posting.doc_id,posting.zone_tfs)
                if tf:scores[posting.doc_id]+=idf*tf/(self.k1+tf)
        for did in scores:
            doc=self.index.doc_metadata[did]
            if doc.get('technology') in technologies:scores[did]*=1+self.hierarchical_boost
            elif doc.get('category') in categories:scores[did]*=1+self.hierarchical_boost/2
        order=sorted(scores,key=lambda did:(-scores[did],did))[:top_k]
        return [dict(self.index.doc_metadata[did],doc_id=did,rank=rank,score=round(scores[did],6)) for rank,did in enumerate(order,1)]
