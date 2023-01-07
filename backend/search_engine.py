from typing import Iterator, List

import numpy as np
import pandas as pd

from backend.pickle_handler import PickleHandler
from backend.query_processor import QueryProcessor

PAGE_RANK_PATH = "page_rank"
PAGE_VIEWS_PATH = "page_views"

INDICES = ['text', 'text_with_stemming', 'title', 'anchor_text']
FILES = ['tf_idf', 'page_rank', 'page_views', 'dl_body', 'dl_title']


class SearchEngine:
    def __init__(self):
        self.pickle_handler = PickleHandler()
        self.query_processor = QueryProcessor()

    def set_indices(self):
        for index_name in INDICES:
            setattr(self, f'{index_name}_index', self.pickle_handler.get_index(index_name))

    def set_dicts(self):
        for file in FILES:
            setattr(self, f'{file}_dict', self.pickle_handler.read_pickle_file(f'{file}.pkl'))

    def get_posting_iter(self, index_name: str):
        """
        This function returning the iterator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        index = getattr(self, f'{index_name}_index')
        words, pls = zip(*index.posting_lists_iter())
        return words, pls

    def get_candidate_documents_and_scores(self, query: List[str], index_name: str, words: Iterator, pls: Iterator):
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
        index = getattr(self, f'{index_name}_index')
        candidates = {}
        for term in np.unique(query):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id, (freq / DL[str(doc_id)]) * np.log10(len(DL) / index.df[term])) for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def search_body(self, query: str, stemming: bool = False):
        processed_query = self.query_processor.process(query, stemming)

    def search_title(self, query: str):
        processed_query = self.query_processor.process(query)

    def search_anchor(self, query: str):
        processed_query = self.query_processor.process(query)


