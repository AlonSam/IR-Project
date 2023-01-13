from typing import Iterator, List

import numpy as np
import pandas as pd

from backend.inverted_index_gcp import InvertedIndex
from backend.pickle_handler import PickleHandler
from backend.query_processor import QueryProcessor
from backend.ranker import Ranker

INDICES = ['text', 'text_with_stemming', 'title', 'anchor_text']
PICKLE_FILES = ['idf_text', 'idf_text_with_stemming', 'page_views', 'dl_text', 'dl_text_with_stemming', 'dl_title',
                'id2title', 'docs_norm']
BUCKET_NAME = 'ofek_alon_project'

class SearchEngine:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.ranker = Ranker()
        self.pickle_handler = PickleHandler()
        self.page_rank = self.pickle_handler.get_page_rank()
        self.set_indices()
        self.set_dicts()

    def set_indices(self):
        for index_name in INDICES:
            setattr(self, f'{index_name}_index', self.pickle_handler.get_index(index_name))
            index = getattr(self, f'{index_name}_index')
            index.name = index_name
            print(f'Successfully read {index_name} index')
            # index.download_posting_locs(BUCKET_NAME, index_name)
            # print(f'Successfully transferred all posting locs for {index_name}')

    def set_dicts(self):
        for file in PICKLE_FILES:
            setattr(self, f'{file}_dict', self.pickle_handler.read_pickle_from_gcp(f'{file}.pkl'))
            print(f'Successfully read {file} dictionary')

    def get_candidate_documents_and_scores(self, query: List[str], index: InvertedIndex, posting_lists):
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
        epsilon = .0000001
        DL = getattr(self, f'dl_{index.name}_dict')
        idf_dict = getattr(self, f'idf_{index.name}_dict')
        candidates = {}
        for term in np.unique(query):
            if term in posting_lists.keys():
                list_of_doc = posting_lists[term]
                if list_of_doc is not None:
                    normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * idf_dict[term]) for
                                       doc_id, freq in list_of_doc]
                    #                 normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * np.log10(index.num_of_docs / (index.df[term] + epsilon))) for
                    #                                doc_id, freq in list_of_doc]
                    for doc_id, tfidf in normlized_tfidf:
                        candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
        return candidates

    def generate_document_tfidf_dict(self, query, index, posting_lists):
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
        candidates_scores = self.get_candidate_documents_and_scores(query, index, posting_lists)
        D = {}
        for (doc_id, term), tfidf in candidates_scores.items():
            if doc_id not in D.keys():
                D[doc_id] = {}
            D[doc_id][term] = tfidf
        return D

    def get_posting_lists(self, index, tokens: List[str]):
        posting_lists = {}
        path = f'postings_gcp_{index.name}/'
        for token in tokens:
            posting_lists[token] = index.get_posting_list(path, token)
        return posting_lists

    def search_body(self, query: str, stemming: bool = False, N: int = 100):
        processed_query = self.query_processor.process(query, stemming)
        index_name = 'text_with_stemming' if stemming is True else 'text'
        index = getattr(self, f'{index_name}_index')
        posting_lists = self.get_posting_lists(index, processed_query)
        Q = self.query_processor.generate_query_tfidf_dict(processed_query, index)
        D = self.generate_document_tfidf_dict(processed_query, index, posting_lists)
        cosine_dict = self.ranker.cosine_similarity(D, Q, self.docs_norm_dict)
        return self.ranker.top_N_documents(cosine_dict, N)

    def search_title(self, query: str):
        processed_query = self.query_processor.process(query)
        index_name = 'title'

    def search_title_binary_ranking(self, query: str) -> List[int]:
        processed_query = self.query_processor.process(query)
        index_name = 'title'
        index = getattr(self, f'{index_name}_index')
        return self.ranker.binary_ranking(processed_query, index)

    def search_anchor(self, query: str) -> List[int]:
        processed_query = self.query_processor.process(query)
        index_name = 'anchor_text'
        index = getattr(self, f'{index_name}_index')
        return self.ranker.binary_ranking(processed_query, index)

    def page_rank(self, wiki_ids: List[int]):
        pr_df = self.pickle_handler.get_page_rank()
        return self.ranker.page_rank(pr_df, wiki_ids)

    def page_views(self, wiki_ids: List[int]):
        return self.ranker.page_views(self.page_views_dict, wiki_ids)



