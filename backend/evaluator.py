from typing import List, Dict

import numpy as np

METRICS = ['Map @K', 'Recall @K', 'Precision @K', 'R Precision', 'Reciprocal Rank @K', 'Fallout Rate', 'F Score']


class Evaluator:
    def __init__(self, num_of_words: int):
        self.num_of_words = num_of_words

    def evaluate(self, true_rank: List[int], predicted_rank: List[int]) -> Dict[str, float]:
        return {
            'Map @K': self.map_at_k(true_rank, predicted_rank),
            'Recall @K': self.recall_at_k(true_rank, predicted_rank),
            'Precision @K': self.precision_at_k(true_rank, predicted_rank),
            'R Precision': self.r_precision(true_rank, predicted_rank),
            'Reciprocal Rank @K': self.reciprocal_rank_at_k(true_rank, predicted_rank),
            'Fallout Rate': self.fallout_rate(true_rank, predicted_rank),
            'F Score': self.f_score(true_rank, predicted_rank)
        }

    @staticmethod
    def _get_num_of_relevant_documents(true_rank: List[int], predicted_rank: List[int], k: int):
        return sum([1 if doc in true_rank else 0 for doc in predicted_rank[:k]])

    @staticmethod
    def _intersect(list1, list2):
        """
        This function perform an intersection between two lists.

        Parameters
        ----------
        list1: list of documents. Each element is a doc_id.
        list2: list of documents. Each element is a doc_id.

        Returns:
        ----------
        list with the intersection (without duplicates) of list1 and list2
        """
        return list(set(list1) & set(list2))

    def recall_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40):
        """
        This function calculate the recall@k metric.

        Parameters
        -----------
        k: integer, a number to slice the length of the predicted_rank

        Returns:
        -----------
        float, recall@k with 3 digits after the decimal point.
        """
        relevant_documents = self._get_num_of_relevant_documents(true_rank, predicted_rank, k)
        return round(relevant_documents / len(true_rank), 3)

    def precision_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40):
        """
        This function calculate the precision@k metric.

        Parameters
        -----------
        k: integer, a number to slice the length of the predicted_rank

        Returns:
        -----------
        float, precision@k with 3 digits after the decimal point.
        """
        relevant_documents = self._get_num_of_relevant_documents(true_rank, predicted_rank, k)
        return round(relevant_documents / k, 3)

    def r_precision(self, true_rank: List[int], predicted_rank: List[int]):
        """
        This function calculate the r-precision metric.

        Returns:
        -----------
        float, r-precision with 3 digits after the decimal point.
        """
        pred_shorter_lst = predicted_rank[:len(true_rank)]
        precision = len(self._intersect(true_rank, pred_shorter_lst))
        return round(precision / len(true_rank), 3)

    @staticmethod
    def reciprocal_rank_at_k(true_rank: List[int], predicted_rank: List[int], k: int = 40):
        """
        This function calculate the reciprocal_rank@k metric.
        Parameters
        -----------
        k: integer, a number to slice the length of the predicted_list

        Returns:
        -----------
        float, reciprocal rank@k with 3 digits after the decimal point.
        """
        for doc in predicted_rank[:k]:
            if doc in true_rank:
                found_at = predicted_rank.index(doc) + 1
                reciprocal_rank_score = round(1 / found_at, 3)
                return reciprocal_rank_score
        return 0

    def fallout_rate(self, true_rank: List[int], predicted_rank: List[int], k: int = 40):
        """
        This function calculate the fallout_rate@k metric.

        Parameters
        -----------
        true_rank: list of relevant documents. Each element is a doc_id.
        predicted_rank: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
        k: integer, a number to slice the length of the predicted_rank

        Returns:
        -----------
        float, fallout_rate@k with 3 digits after the decimal point.
        """
        k_predicted_lst = predicted_rank[:k]
        intersection_len = len(self._intersect(true_rank, k_predicted_lst))
        not_relevant_words = self.num_of_words - len(true_rank)
        return round((len(k_predicted_lst) - intersection_len) / not_relevant_words, 3)

    def f_score(self, true_rank: List[int], predicted_rank: List[int], k: int = 40):
        """
        This function calculate the f_score@k metric.

        Parameters
        -----------
        k: integer, a number to slice the length of the predicted_list

        Returns:
        -----------
        float, f-score@k with 3 digits after the decimal point.
        """
        precision_at_k = self.precision_at_k(true_rank, predicted_rank, k)
        recall_at_k = self.recall_at_k(true_rank, predicted_rank, k)
        if precision_at_k == 0 or (recall_at_k == 0 and precision_at_k == 0):
            return 0
        f_measure = (2 * precision_at_k) / (precision_at_k + recall_at_k)
        return round(f_measure, 3)


    def map_at_k(self, true_rank: List[int], predicted_rank: List[int], k: int = 40):
        precisions = []
        for i in range(k):
            if true_rank[i] == predicted_rank[i]:
                precisions.append(self.precision_at_k(true_rank, predicted_rank, i + 1))
        return np.mean(precisions)