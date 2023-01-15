import time
from typing import List
import json

import numpy as np
from sklearn.model_selection import ShuffleSplit

from evaluator import Evaluator

file_name = 'new_train.json'


class GridSearchCV:
    def __init__(self, train_size: float = 0.8):
        self.train_size = train_size
        with open(file_name) as json_file:
            self.train_data = list(json.load(json_file).items())
        self.evaluator = Evaluator()
        self.best_params = {}

    def train_test_split(self, cv: int = 5):
        shuffle_split = ShuffleSplit(n_splits=cv, train_size=self.train_size, random_state=0)
        trains = []
        tests = []
        for i, (train_index, test_index) in enumerate(shuffle_split.split(self.train_data)):
            trains.append(list(self.train_data[ind] for ind in train_index))
            tests.append(list(self.train_data[ind] for ind in test_index))
        return trains, tests

    def grid_search_cv(self,
                       search_func,
                       cv: int = 5,
                       num_results: List[int] = [100],
                       expand: List[bool] = [False],
                       similar_words: List[int] = [6],
                       min_similarity: List[float] = [0.7],
                       stemming: List[bool] = [False],
                       ):
        trains, tests = self.train_test_split(cv=cv)
        best_params = []
        best_map = 0.0
        for train, test in zip(trains, tests):
            for num_result in num_results:
                for to_expand in expand:
                    for similar_word in similar_words:
                        for similarity in min_similarity:
                            for stem in stemming:
                                scores = []
                                times = []
                                for query_tup in train:
                                    query, true_results = query_tup
                                    start = time.time()
                                    predicted_results = search_func(query,
                                                                    N=num_result,
                                                                    expand=to_expand,
                                                                    similar_words=similar_word,
                                                                    similarity=similarity,
                                                                    stemming=stem)
                                    res = [doc_id for (doc_id, title) in predicted_results]
                                    t = time.time() - start
                                    times.append(t)
                                    scores.append(self.evaluator.map_at_k(true_results, res))
                                average_score = np.mean(scores)
                                average_time = np.mean(times)
                                if average_score > best_map:
                                    best_params.append(
                                        {
                                        'num_results': num_result,
                                        'expand': to_expand,
                                        'similar_words': similar_word,
                                        'min_similarity': similarity,
                                        'stemming': stem,
                                        'average_time': average_time
                                        }
                                    )
        for param in best_params[0].keys():
            self.best_params[param] = sum(d[param] for d in best_params) / len(best_params)