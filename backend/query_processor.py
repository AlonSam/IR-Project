from collections import Counter
from typing import List, Dict
import numpy as np
from preprocess.stemmer import Stemmer
from preprocess.tokenizer import Tokenizer
import gensim.downloader as api


class QueryProcessor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.stemmer = Stemmer()
        self.word2vec_model = api.load("glove-wiki-gigaword-300")

    def process(self, query: str,
                stemming: bool = False,
                expand: bool = False,
                similar_words: int = 6,
                similarity: float = 0.7) -> List[str]:
        tokens = self.tokenizer.tokenize(query)
        if stemming is True:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if expand is True:
            tokens += self._get_similar_words(tokens, similar_words, similarity)
        return tokens

    def re_process(self,
                   query: str,
                   stemming: bool = False) -> List[str]:
        tokens = self.tokenizer.re_tokenize(query)
        if stemming is True:
            tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def generate_query_tfidf_dict(self, query: List[str], index) -> Dict[str, float]:
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
        Q = {}
        counter = Counter(query)
        for ind, token in enumerate(np.unique(query)):
            if token in index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = np.log10(index.num_of_docs / (df + epsilon))  # smoothing
                Q[token] = tf * idf
        return Q

    def _get_similar_words(self,
                           query: List[str],
                           similar_words: int = 6,
                           min_similarity: float = 0.7) -> List[str]:
        n_most_similar = []
        for token in query:
            try:
                n_most_similar += self.word2vec_model.most_similar(positive=[token], topn=similar_words)
            except:
                continue
        res = []
        for word, similarity in n_most_similar:
            if similarity > min_similarity:
                tokens = self.tokenizer.tokenize(word)
                if len(tokens) > 0:
                    res.append(tokens[0])
        return res