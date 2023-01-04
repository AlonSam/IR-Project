import re
from collections import Counter
from typing import List, Tuple
from nltk import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords


class Tokenizer:
    def __init__(self):
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
        self.nltk_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"] # TODO
        self.stopwords = self.nltk_stopwords.union(self.corpus_stopwords)
        self.stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer(language='english')

    def tokenize(self, text: str, doc_id, stemming: bool = False) -> List[Tuple]:
        tokens = self._get_tokens(text)
        counter = self._word_count(tokens, doc_id, stemming)
        return counter

    def _get_tokens(self, text: str) -> List[str]:
        tokens = [token.group() for token in self.RE_WORD.finditer(text.lower())]
        return [token for token in tokens if token not in self.stopwords]

    def _word_count(self, tokens: List[str], doc_id: int, stemming: bool) -> List[Tuple]:
        ''' Count the frequency of each word in `text` (tf) that is not included in
        `all_stopwords` and return entries that will go into our posting lists.
        Parameters:
        -----------
          text: str
            Text of one document
          id: int
            Document id
        Returns:
        --------
          List of tuples
            A list of (token, (doc_id, tf)) pairs
            for example: [("Anarchism", (12, 5)), ...]
        '''
        if stemming:
            counter = Counter([self.snowball_stemmer.stem(token) for token in tokens])
        else:
            counter = Counter(tokens)
        return [(token, (doc_id, count)) for (token, count) in counter.items()]
