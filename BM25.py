import math
from collections import defaultdict
from typing import List

from ranker import Ranker


class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """
    def __init__(self, index, dl_dict, anchor_index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.anchor_index = anchor_index
        self.dl_dict = dl_dict
        self.N = len(dl_dict)
        self.AVGDL = sum(dl_dict.values()) / self.N
        self.ranker = Ranker()

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query: List[str], N: int = 3, add_anchor: bool = False):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        idf = self.calc_idf(query)
        posting_lists = self.ranker.get_posting_lists(self.index, query)
        self.index.posting_lists.update(posting_lists)
        if add_anchor is True:
            anchor_posting_lists = self.ranker.get_posting_lists(self.anchor_index, query)
            candidates = self.ranker.get_candidate_documents_for_term(query, posting_lists, anchor_posting_lists)
        else:
            candidates = self.ranker.get_candidate_documents_for_term(query, posting_lists)
        return sorted([(doc_id, score) for (doc_id, score) in self._score(candidates, idf).items()], key=lambda x: x[1], reverse=True)[:N]

    def _score(self, candidates, idf):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        scores = defaultdict(float)
        for term, docs in candidates.items():
            if term in self.index.term_total.keys():
                for doc_id, tf in docs.items():
                    doc_len = self.dl_dict[doc_id]
                    numerator = idf[term] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    scores[doc_id] += numerator / denominator
        return scores