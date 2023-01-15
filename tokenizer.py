import re
from typing import List, FrozenSet
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Tokenizer:
    def __init__(self):
        self.nltk_stopwords = frozenset(stopwords.words('english'))
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)

    def _get_stopwords(self) -> FrozenSet[str]:
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]  # TODO
        return self.nltk_stopwords.union(corpus_stopwords)

    def tokenize(self, text: str) -> List[str]:
        stopwords = self._get_stopwords()
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in stopwords and token.isalpha()]

    def re_tokenize(self, text: str) -> List[str]:
        stopwords = self._get_stopwords()
        tokens = [token.group() for token in self.RE_WORD.finditer(text.lower())]
        return [token for token in tokens if token not in stopwords]
