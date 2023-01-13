from collections import defaultdict
from typing import List, Dict
import numpy as np
import pandas as pd
from numpy.linalg import norm

from backend.inverted_index_gcp import InvertedIndex


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def cosine_similarity(D, Q, docs_norm):
        query_norm = norm(list(Q.values()))
        cosine = defaultdict(int)
        for doc_id, term_dict in D.items():
            for term, tfidf in term_dict.items():
                cosine[doc_id] += (tfidf * Q[term])
        for doc_id, sim in cosine.items():
            cosine[doc_id] = sim / (docs_norm[doc_id] * query_norm)
        return cosine

    @staticmethod
    def top_N_documents(cosine_dict, N=100):
        """
        This function sort and filter the top N docuemnts (by score) for each query.

        Parameters
        ----------
        df: DataFrame (queries as rows, documents as columns)
        N: Integer (how many document to retrieve for each query)

        Returns:
        ----------
        top_N: dictionary is the following stracture:
              key - query id.
              value - sorted (according to score) list of pairs lengh of N. Eac pair within the list provide the following information (doc id, score)
        """
        sorted_scores = sorted([(doc_id, scores) for (doc_id, scores) in cosine_dict.items()], key=lambda x: x[1],
                               reverse=True)[:N]
        return [doc_id for (doc_id, score) in sorted_scores]

    @staticmethod
    def binary_ranking(query: List[str], index: InvertedIndex):
        relevant_docs = defaultdict(int)
        path = f'postings_gcp_{index.name}/'
        for word in query:
            pls = index.get_posting_list(path, word)
            for doc_id, _ in pls:
                relevant_docs[doc_id] += 1
        ranks = sorted(list(relevant_docs.items()), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in ranks]

    def page_rank(self, df: pd.DataFrame, wiki_ids: List[int]) -> List[float]:
        ranks = []
        for wiki_id in wiki_ids:
            row = df[df['id'] == wiki_id].to_dict('records')
            ranks.append(row[0]['page_rank'])
        return ranks

    def page_views(self, page_views_dict: Dict[int, int], wiki_ids: List[int]) -> List[int]:
        views = []
        for wiki_id in wiki_ids:
            views.append(page_views_dict[wiki_id])
        return views