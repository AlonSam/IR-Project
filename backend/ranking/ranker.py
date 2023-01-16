from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from numpy.linalg import norm
from itertools import groupby
from backend.inverted_index_gcp import InvertedIndex
from sklearn.preprocessing import MinMaxScaler


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def cosine_similarity(D: Dict[int, Dict[str, float]], Q: Dict[str, float], docs_norm: Dict[int, int]) -> float:
        """
        Calculates the cosine similarity between documents and a query
        :param D: dictionary that maps each document to a dictionary that maps each term to its tf-idf value in the document.
        :param Q: dictionary that maps each term to its tf-idf value in the query.
        :param docs_norm: dictionary that maps each document to its norm.
        :return: dictionary that maps each document to its cosine similarity sore.
        """
        query_norm = norm(list(Q.values()))
        cosine = defaultdict(int)
        for doc_id, term_dict in D.items():
            for term, tfidf in term_dict.items():
                cosine[doc_id] += (tfidf * Q[term])
        for doc_id, sim in cosine.items():
            cosine[doc_id] = sim / (docs_norm[doc_id] * query_norm)
        return cosine

    @staticmethod
    def top_N_documents(cosine_dict: Dict[int, float], N: int = 100):
        """
        This function sort and filter the top N documents (by score) for each query.

        Parameters
        ----------
        cosine_dict: Dictionary (doc_ids as keys, cosine similary as values)
        N: Integer (how many document to retrieve for each query)

        Returns:
        ----------
        top_N: dictionary is the following structure:
              key - doc id.
              value - sorted (according to score) list of pairs length of N. Each pair within the list provide the following information (doc id, score)
        """
        return sorted([(doc_id, scores) for (doc_id, scores) in cosine_dict.items()], key=lambda x: x[1],
                               reverse=True)[:N]

    @staticmethod
    def binary_ranking(query: List[str], index: InvertedIndex) -> List[int]:
        """
        Returns a list of document ids sorted by their binary ranking score, for a given query.
        """
        relevant_docs = defaultdict(int)
        path = f'postings_gcp_{index.name}/'
        for word in query:
            if word in index.posting_lists.keys():
                pls = index.posting_lists[word]
            else:
                bins = [loc[0] for loc in index.posting_locs[word]]
                index.download_posting_locs_for_query(path, bins)
                pls = index.get_posting_list(path, word)
                if pls is None:
                    return []
            for doc_id, _ in pls:
                relevant_docs[doc_id] += 1
        ranks = sorted(list(relevant_docs.items()), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in ranks]

    def page_rank(self, df: pd.DataFrame, wiki_ids: List[int]) -> List[float]:
        """
        Returns the page rank of the given wiki_ids.
        """
        ranks = []
        for wiki_id in wiki_ids:
            row = df[df['id'] == wiki_id].to_dict('records')
            try:
                ranks.append(row[0]['page_rank'])
            except:
                ranks.append(0)
        return ranks

    def page_views(self, page_views_dict: Dict[int, int], wiki_ids: List[int]) -> List[int]:
        """
        Returns the number of page views of the given wiki_ids.
        """
        views = []
        for wiki_id in wiki_ids:
            try:
                views.append(page_views_dict[wiki_id])
            except:
                views.append(0)
        return views

    @staticmethod
    def get_candidate_documents_and_scores(query: List[str], posting_lists, dl_dict, idf_dict):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query):
            if term in posting_lists.keys():
                list_of_doc = posting_lists[term]
                if list_of_doc is not None:
                    normlized_tfidf = [(doc_id, (freq / dl_dict[doc_id]) * idf_dict[term]) for
                                       doc_id, freq in list_of_doc]
                    for doc_id, tfidf in normlized_tfidf:
                        candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
        return candidates

    def generate_document_tfidf_dict(self, query, index, posting_lists, dl_dict, idf_dict):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.


        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """
        candidates_scores = self.get_candidate_documents_and_scores(query, posting_lists, dl_dict, idf_dict)
        D = {}
        for (doc_id, term), tfidf in candidates_scores.items():
            if doc_id not in D.keys():
                D[doc_id] = {}
            D[doc_id][term] = tfidf
        return D

    @staticmethod
    def get_candidate_documents(query: List[str], posting_lists):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = []
        for term in np.unique(query):
            if term in posting_lists.keys():
                candidates += [doc_id for (doc_id, tf) in posting_lists[term]]
        return np.unique(candidates)

    @staticmethod
    def get_candidate_documents_for_term(query: List[str], posting_lists, anchor_posting_lists={}, add_anchor: bool = False):
        """
        Returns a list of document ids that are candidates for a given query.
        """
        candidates = {}
        for term in np.unique(query):
            if term in posting_lists.keys():
                try:
                    candidates[term] = {doc_id: tf for (doc_id, tf) in posting_lists[term] if tf > 0}
                    if add_anchor is True and term in anchor_posting_lists.keys():
                        candidates[term].update({doc_id: tf for (doc_id, tf) in anchor_posting_lists[term] if doc_id not in candidates[term].keys()})
                except:
                    continue
        return candidates

    @staticmethod
    def get_posting_lists(index: InvertedIndex, tokens: List[str]):
        """
        Given an index and a list of tokens, returns the posting lists of the tokens.
        """
        posting_lists = {}
        path = f'postings_gcp_{index.name}/'
        for token in tokens:
            if token in index.posting_lists.keys():
                posting_lists[token] = index.posting_lists[token]
            else:
                try:
                    bins = [loc[0] for loc in index.posting_locs[token]]
                    index.download_posting_locs_for_query(path, bins)
                    posting_lists[token] = index.get_posting_list(path, token)
                except:
                    continue
        return posting_lists

    def merge_results(self, scores1,  scores2, w1=0.5, w2=0.5, N=20):
        """
        This function merge and sort documents retrieved by its weighted score (e.g., title and body).

        Parameters:
        -----------
        title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)

        body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)
        title_weight: float, for weigted average utilizing title and body scores
        text_weight: float, for weigted average utilizing title and body scores
        N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        dictionary of querires and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id,score).
        """
        scores1_std = self._min_max_scale(scores1)
        scores2_std = self._min_max_scale(scores2)
        new_score = [(doc_id, w1 * score) for (doc_id, score) in scores1_std]
        new_score += [(doc_id, w2 * score) for (doc_id, score) in scores2_std]
        g_list = [list(g) for k, g in groupby(sorted(new_score), lambda x: x[0])]
        merged_scores = sorted(
            [(i[0], j[0] + j[1]) if len(j) > 1 else (i[0], j[0]) for i, j in [zip(*i) for i in g_list]],
            key=lambda x: x[1], reverse=True)[:N]
        return merged_scores

    def merge_title_page_rank_views(self,
                                    title_scores: List[Tuple[int, float]],
                                    page_rank_scores: List[int],
                                    page_views_scores: List[int],
                                    w_title: float = 0.5,
                                    w_page_rank: float = 0.25,
                                    w_page_views: float = 0.25,
                                    N: int = 10):
        """
        :param title_scores: A list containing doc_ids and scores returned by the title search
        :param page_rank_scores: A list containing doc_ids and scores returned by page rank
        :param page_views_scores: A list containing doc_ids and scores returned by page views
        :param w_title: weight for title.
        :param w_page_rank: weight for page rank.
        :param w_page_views: weight for page views.
        :param N: Number of results to return
        :return: A merged ranking of 3 search methods
        """
        try:
            title_scores_std = self._min_max_scale(title_scores)
            page_rank_scores_std = self._min_max_scale(page_rank_scores)
            page_views_scores_std = self._min_max_scale(page_views_scores)
            new_score = [(doc_id, w_title * score) for (doc_id, score) in title_scores_std]
            new_score += [(doc_id, w_page_rank * score) for (doc_id, score) in page_rank_scores_std]
            new_score += [(doc_id, w_page_views * score) for (doc_id, score) in page_views_scores_std]
            g_list = [list(g) for k, g in groupby(sorted(new_score), lambda x: x[0])]
            merged_scores = []
            for i, j in [zip(*i) for i in g_list]:
                if len(j) == 3:
                    merged_scores.append((i[0], j[0] + j[1] + j[2]))
                elif len(j) == 2:
                    merged_scores.append((i[0], j[0] + j[1]))
                else:
                    merged_scores.append((i[0], j[0]))
            return sorted(merged_scores, key=lambda x: x[1], reverse=True)[:N]
        except:
            return []

    def merge_all(self,
                                    body_scores: List[Tuple[int, float]],
                                    title_scores: List[Tuple[int, float]],
                                    page_rank_scores: List[int],
                                    page_views_scores: List[int],
                                    w_body: float = 0.25,
                                    w_title: float = 0.25,
                                    w_page_rank: float = 0.25,
                                    w_page_views: float = 0.25,
                                    N: int = 10):
        try:
            body_scores_std = self._min_max_scale(body_scores)
            title_scores_std = self._min_max_scale(title_scores)
            page_rank_scores_std = self._min_max_scale(page_rank_scores)
            page_views_scores_std = self._min_max_scale(page_views_scores)
            new_score = [(doc_id, w_body * score) for (doc_id, score) in body_scores_std]
            new_score += [(doc_id, w_title * score) for (doc_id, score) in title_scores_std]
            new_score += [(doc_id, w_page_rank * score) for (doc_id, score) in page_rank_scores_std]
            new_score += [(doc_id, w_page_views * score) for (doc_id, score) in page_views_scores_std]
            g_list = [list(g) for k, g in groupby(sorted(new_score), lambda x: x[0])]
            merged_scores = []
            for i, j in [zip(*i) for i in g_list]:
                if len(j) == 4:
                    merged_scores.append((i[0], j[0] + j[1] + j[2] + j[3]))
                elif len(j) == 3:
                    merged_scores.append((i[0], j[0] + j[1] + j[2]))
                elif len(j) == 2:
                    merged_scores.append((i[0], j[0] + j[1]))
                else:
                    merged_scores.append((i[0], j[0]))
            return sorted(merged_scores, key=lambda x: x[1], reverse=True)[:N]
        except:
            return []

    @staticmethod
    def _min_max_scale(doc_id_scores):
        scaler = MinMaxScaler()
        scores = np.array([score for (doc_id, score) in doc_id_scores]).reshape(-1, 1)
        scores_std = scaler.fit_transform(scores)[:, 0]
        return [(doc_id, score_std) for ((doc_id, score), score_std) in zip(doc_id_scores, scores_std)]