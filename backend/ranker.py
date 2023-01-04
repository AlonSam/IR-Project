from typing import List

import numpy as np


class Ranker:
    def __init__(self):

    def cosine_similarity(self, document: List[str], query: List[str]):
        cosine = np.dot(document, query) / (np.linalg.norm(document, axis=1) * np.linalg.norm(query))
        return dict(enumerate(cosine))

    def binary_ranking(self):
        pass

    def page_rank(self):
        pass

    def page_views(self):
        pass