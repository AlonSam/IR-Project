from nltk import SnowballStemmer


class Stemmer:
    def __init__(self):
        self.stemmer = SnowballStemmer(language='english')

    def stem(self, word: str) -> str:
        return self.stemmer.stem(word)