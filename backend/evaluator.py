from typing import List, Dict
import numpy as np


class Evaluator:
    def __init__(self, num_of_words: int = 0):
        self.num_of_words = num_of_words

    def evaluate(self,
                 true_rank: List[int],
                 predicted_rank: List[int],
                 k: int = 40) -> Dict[str, float]:
        """
        Evaluates the predicted ranks vs the true ranks.
        :param true_rank: List of true ranks
        :param predicted_rank: List of predicted ranks
        :param k: int.
        :return:
        Dictionary of scores for different evaluation metrics
        """
        return {
            'Map @K': self.map_at_k(true_rank, predicted_rank, k),
            'Average Precision@K': self.map_at_k(true_rank, predicted_rank, k),
            'Recall @K': self.recall_at_k(true_rank, predicted_rank, k),
            'Precision @K': self.precision_at_k(true_rank, predicted_rank, k),
            'R Precision': self.r_precision(true_rank, predicted_rank),
            'Reciprocal Rank @K': self.reciprocal_rank_at_k(true_rank, predicted_rank, k),
            'Fallout Rate': self.fallout_rate(true_rank, predicted_rank, k),
            'F Score': self.f_score(true_rank, predicted_rank, k)
        }

    @staticmethod
    def _get_num_of_relevant_documents(true_rank: List[int], predicted_rank: List[int], k: int) -> int:
        """
        returns the number of documents in the first k results of predicted_rank that also appear in true_rank
        """
        return sum([1 if doc in true_rank else 0 for doc in predicted_rank[:k]])

    @staticmethod
    def _intersect(list1, list2):
        """
        Returns the intersection of two lists.
        """
        return list(set(list1) & set(list2))

    def recall_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the recall@k metric.
        """
        relevant_documents = self._get_num_of_relevant_documents(true_rank, predicted_rank, k)
        return round(relevant_documents / len(true_rank), 3)

    def precision_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the precision@k metric.
        """
        relevant_documents = self._get_num_of_relevant_documents(true_rank, predicted_rank, k)
        return round(relevant_documents / k, 3)

    def r_precision(self, true_rank: List[int], predicted_rank: List[int]) -> float:
        """
        Calculate the r-precision metric
        """
        precision = len(self._intersect(true_rank, predicted_rank[:len(true_rank)]))
        return round(precision / len(true_rank), 3)

    @staticmethod
    def reciprocal_rank_at_k(true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the reciprocal_rank@k metric.
        """
        for doc in predicted_rank[:k]:
            if doc in true_rank:
                found_at = predicted_rank.index(doc) + 1
                reciprocal_rank_score = round(1 / found_at, 3)
                return reciprocal_rank_score
        return 0.0

    def fallout_rate(self, true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the fallout_rate@k metric.
        """
        k_predicted_lst = predicted_rank[:k]
        intersection_len = len(self._intersect(true_rank, k_predicted_lst))
        not_relevant_words = self.num_of_words - len(true_rank)
        return round((len(k_predicted_lst) - intersection_len) / not_relevant_words, 3)

    def f_score(self, true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the f_score@k metric.
        """
        precision_at_k = self.precision_at_k(true_rank, predicted_rank, k)
        recall_at_k = self.recall_at_k(true_rank, predicted_rank, k)
        if precision_at_k == 0 or (recall_at_k == 0 and precision_at_k == 0):
            return 0
        f_measure = (2 * precision_at_k) / (precision_at_k + recall_at_k)
        return round(f_measure, 3)

    def map_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40) -> float:
        """
        Calculates the MAP@K metric.
        """
        precisions = []
        for i, doc_id in enumerate(predicted_rank[:k]):
            if doc_id in true_rank:
                precisions.append(self.precision_at_k(true_rank, predicted_rank, i + 1))
        return round(np.mean(precisions), 3) if len(precisions) > 0 else 0.0