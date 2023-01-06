import re
from typing import List, FrozenSet

from nltk.corpus import stopwords


class Tokenizer:
    def __init__(self):
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)

    @staticmethod
    def _get_stopwords() -> FrozenSet[str]:
        nltk_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]  # TODO
        return nltk_stopwords.union(corpus_stopwords)

    def tokenize(self, text: str) -> List[str]:
        stopwords = self._get_stopwords()
        tokens = [token.group() for token in self.RE_WORD.finditer(text.lower())]
        return [token for token in tokens if token not in stopwords]
