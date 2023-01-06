import math
from collections import Counter

import numpy as np

from backend.preprocess.stemmer import Stemmer
from backend.preprocess.tokenizer import Tokenizer


class QueryProcessor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.stemmer = Stemmer()

    def process(self, query: str, stemming: bool = False):
        tokens = self.tokenizer.tokenize(query)
        if stemming is True:
            processed_query = [self.stemmer.stem(token) for token in tokens]


    def generate_query_tfidf_vector(self, query, index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001
        total_vocab_size = len(index.term_total)
        Q = np.zeros((total_vocab_size))
        term_vector = list(index.term_total.keys())
        counter = Counter(query)
        for token in np.unique(query):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = np.log10(index.num_of_docs / (df + epsilon))  # smoothing
                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q