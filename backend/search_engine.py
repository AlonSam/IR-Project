from typing import List, Tuple

from ranking.BM25 import BM25
from pickle_handler import PickleHandler
from query_processor import QueryProcessor
from ranking.ranker import Ranker

INDICES = ['text', 'text_with_stemming', 'title', 'title_with_stemming', 'anchor_text']
PICKLE_FILES = ['idf_text', 'idf_text_with_stemming', 'page_views', 'dl_text', 'dl_text_with_stemming', 'dl_title',
                'dl_title_with_stemming', 'id2title', 'docs_norm']
BUCKET_NAME = 'ofek_alon_project'


class SearchEngine:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.ranker = Ranker()
        self.pickle_handler = PickleHandler()
        self.page_rank_df = self.pickle_handler.get_page_rank()
        self.set_indices()
        self.set_dicts()

    def set_indices(self) -> None:
        for index_name in INDICES:
            setattr(self, f'{index_name}_index', self.pickle_handler.get_index(index_name))
            index = getattr(self, f'{index_name}_index')
            index.name = index_name
            index.posting_lists = {}
            print(f'Successfully read {index_name} index')

    def set_dicts(self) -> None:
        for file in PICKLE_FILES:
            setattr(self, f'{file}_dict', self.pickle_handler.read_pickle_from_gcp(f'{file}.pkl'))
            print(f'Successfully read {file} dictionary')

    def search_body(self, query: str,
                    stemming: bool = False,
                    N: int = 100,
                    type: str = 'tfidf',
                    expand: bool = False,
                    similar_words: int = 6,
                    similarity: float = 0.7) -> List[Tuple[int, str]]:
        """
        Given a query, uses the body of documents to retrieve relevant documents.
        :param query: str
        :param stemming: bool. Whether to use text_index or text_with_stemming_index
        :param N: int. The number of results to return.
        :param type: Which search method to use.
        :return:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
        """
        if type == 'tfidf':
            ranks = self.search_body_tfidf(query, stemming=stemming, N=N)
        else:
            ranks = self.search_body_BM25(query, stemming=stemming, N=N, expand=expand, similar_words=similar_words,
                                          similarity=similarity)
        return [(doc_id, self.id2title_dict[doc_id]) for (doc_id, score) in ranks]

    def search_body_tfidf(self,
                          query: str,
                          stemming: bool = False,
                          N: int = 100) -> List[Tuple[int, float]]:
        """
        Given a query, uses TFIDF on the body of documents to retrieve relevant documents.
        :param query: str
        :param stemming: bool. Whether to use text_index or text_with_stemming_index
        :param N: int. The number of results to return.
        :return:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
        """
        processed_query = self.query_processor.re_process(query, stemming)
        index_name = 'text_with_stemming' if stemming is True else 'text'
        index = getattr(self, f'{index_name}_index')
        dl_dict = getattr(self, f'dl_{index_name}_dict')
        idf_dict = getattr(self, f'idf_{index.name}_dict')
        posting_lists = self.ranker.get_posting_lists(index, processed_query)
        index.posting_lists.update(posting_lists)
        Q = self.query_processor.generate_query_tfidf_dict(processed_query, index)
        D = self.ranker.generate_document_tfidf_dict(processed_query, index, posting_lists, dl_dict, idf_dict)
        cosine_dict = self.ranker.cosine_similarity(D, Q, self.docs_norm_dict)
        return self.ranker.top_N_documents(cosine_dict, N)

    def search_body_BM25(self,
                         query: str,
                         k1: float = 2.0,
                         b: float = 0.75,
                         stemming: bool = False,
                         N: int = 100,
                         expand: bool = False,
                         similar_words: int = 6,
                         similarity: float = 0.7,
                         add_anchor: bool = False) -> List[Tuple[int, float]]:
        """
        Given a query, uses BM25 on the body of documents to retrieve relevant documents.
        :param b: float, default: 0.75
        :param k1: float, default: 2.0
        :param query: str
        :param stemming: bool. Whether to use text_index or text_with_stemming_index
        :param N: int. The number of results to return.
        :return:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
        """
        processed_query = self.query_processor.process(query, stemming, expand, similar_words=similar_words,
                                                       similarity=similarity)
        index_name = 'text_with_stemming' if stemming is True else 'text'
        index = getattr(self, f'{index_name}_index')
        dl_dict = getattr(self, f'dl_{index_name}_dict')
        bm25 = BM25(index=index, dl_dict=dl_dict, k1=k1, b=b, anchor_index=self.anchor_text_index)
        return bm25.search(processed_query, N, add_anchor=add_anchor)

    def search_title(self,
                     query: str,
                     stemming: bool = False,
                     N: int = 100, type: str = 'binary',
                     expand: bool = False,
                     similar_words: int = 6,
                     similarity: float = 0.7,
                     add_anchor: bool = False) -> List[Tuple[int, str]]:
        """
        Given a query, uses the title of documents to retrieve relevant documents.
        :param query: str
        :param stemming: bool. Whether to use title_index or title_with_stemming_index
        :param N: int. The number of results to return.
        :param type: Which search method to use.
        :return:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
        """
        if type == 'binary':
            return self.search_title_binary_ranking(query)
        else:
            ranks = self.search_title_BM25(query, stemming=stemming, N=N, expand=expand, similar_words=similar_words,
                                           similarity=similarity, add_anchor=add_anchor)
            return [(doc_id, self.id2title_dict[doc_id]) for (doc_id, score) in ranks]

    def search_title_BM25(self,
                          query: str,
                          k1: float = 2.0,
                          b: float = 0.75,
                          stemming: bool = False,
                          N: int = 100,
                          expand: bool = False,
                          similar_words: int = 6,
                          similarity: float = 0.7,
                          add_anchor: bool = False) -> List[Tuple[int, float]]:
        """
        Given a query, uses BM25 on the title of documents to retrieve relevant documents.
        :param b: float, default: 0.75
        :param k1: float, default: 2.0
        :param query: str
        :param stemming: bool. Whether to use title_index or title_with_stemming_index
        :param N: int. The number of results to return.
        :return:
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, score).
        """
        processed_query = self.query_processor.process(query, stemming, expand, similar_words=similar_words,
                                                       similarity=similarity)
        index_name = 'title_with_stemming' if stemming is True else 'title'
        index = getattr(self, f'{index_name}_index')
        dl_dict = getattr(self, f'dl_{index_name}_dict')
        bm25 = BM25(index=index, dl_dict=dl_dict, k1=k1, b=b, anchor_index=self.anchor_text_index)
        return bm25.search(processed_query, N, add_anchor=add_anchor)

    def search_body_and_title_BM25(self,
                                   query: str,
                                   stemming: bool = False,
                                   N: int = 100,
                                   w1: float = 0.5,
                                   w2: float = 0.5,
                                   expand: bool = False,
                                   similar_words: int = 6,
                                   similarity: float = 0.7,
                                   add_anchor: bool = False) -> List[Tuple[int, str]]:
        body_scores = self.search_body_BM25(query, stemming=stemming, N=N, expand=expand, similar_words=similar_words,
                                            similarity=similarity, add_anchor=add_anchor)
        title_scores = self.search_title_BM25(query, stemming=stemming, N=N, expand=expand, similar_words=similar_words,
                                              similarity=similarity, add_anchor=add_anchor)
        if len(body_scores) == 0 and len(title_scores) == 0:
            return []
        merged_scores = self.ranker.merge_results(body_scores, title_scores, w1=w1, w2=w2)
        return [(doc_id, self.id2title_dict[doc_id]) for (doc_id, score) in merged_scores]

    def search_body_tfidf_and_title_BM25(self,
                                         query: str,
                                         stemming: bool = False,
                                         N: int = 100,
                                         w1: float = 0.5,
                                         w2: float = 0.5,
                                         expand: bool = False,
                                         similar_words: int = 6,
                                         similarity: float = 0.7,
                                         add_anchor: bool = False) -> List[Tuple[int, str]]:
        body_scores = self.search_body_tfidf(query, stemming=stemming, N=N)
        title_scores = self.search_title_BM25(query, stemming=stemming, N=N, expand=expand, similar_words=similar_words,
                                              similarity=similarity, add_anchor=add_anchor)
        if len(body_scores) == 0 and len(title_scores) == 0:
            return []
        merged_scores = self.ranker.merge_results(body_scores, title_scores, w1=w1, w2=w2)
        return [(doc_id, self.id2title_dict[doc_id]) for (doc_id, score) in merged_scores]

    def search_title_binary_ranking(self, query: str) -> List[Tuple[int, str]]:
        """
        Given a query, uses binary ranking on the title of documents to retrieve relevant documents.
        :param query: str
        :return:
        list ALL search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
        """
        processed_query = self.query_processor.re_process(query)
        index_name = 'title'
        index = getattr(self, f'{index_name}_index')
        ranks = self.ranker.binary_ranking(processed_query, index)
        return [(doc_id, self.id2title_dict[doc_id]) for doc_id in ranks]

    def ultimate_search(self,
                        query,
                        k_body: float = 1.5,
                        b_body: float = 0.2,
                        k_title: float = 1.5,
                        b_title: float = 0.2,
                        stemming: bool = True,
                        add_anchor: bool = True,
                        expand: bool = False,
                        w_body: float = 0.8,
                        w_title: float = 0.8,
                        w_page_rank: float = 0.5,
                        w_page_views: float = 0.5,
                        N: int = 5) -> List[Tuple[int, str]]:
        body_scores = self.search_body_BM25(query,
                                            k1=k_body,
                                            b=b_body,
                                            stemming=stemming,
                                            expand=expand,
                                            add_anchor=add_anchor)
        title_scores = self.search_title_BM25(query,
                                              k1=k_title,
                                              b=b_title,
                                              stemming=stemming,
                                              expand=expand,
                                              add_anchor=add_anchor)
        wiki_ids = list(set([doc_id for (doc_id, score) in title_scores + body_scores]))
        page_rank_scores = [(wiki_id, score) for (wiki_id, score) in zip(wiki_ids, self.page_rank(wiki_ids))]
        page_views_scores = [(wiki_id, score) for (wiki_id, score) in zip(wiki_ids, self.page_views(wiki_ids))]
        merged_scores = self.ranker.merge_all(body_scores,
                                              title_scores,
                                              page_rank_scores,
                                              page_views_scores,
                                              w_body=w_body,
                                              w_title=w_title,
                                              w_page_rank=w_page_rank,
                                              w_page_views=w_page_views,
                                              N=N)
        return [(doc_id, self.id2title_dict[doc_id]) for (doc_id, score) in merged_scores]

    def search_anchor(self, query: str) -> List[Tuple[int, str]]:
        """
        Given a query, uses binary ranking on the anchor text of documents to retrieve relevant documents.
        :param query: str
        :return:
        list ALL search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
        """
        processed_query = self.query_processor.re_process(query)
        index_name = 'anchor_text'
        index = getattr(self, f'{index_name}_index')
        ranks = self.ranker.binary_ranking(processed_query, index)
        return [(doc_id, self.id2title_dict[doc_id]) for doc_id in ranks if doc_id in self.id2title_dict.keys()]

    def page_rank(self, wiki_ids: List[int]) -> List[float]:
        """
        Given a list of wiki_ids, returns the page rank of each id
        :param wiki_ids: List of wiki ids
        :return:
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
        """
        return self.ranker.page_rank(self.page_rank_df, wiki_ids)

    def page_views(self, wiki_ids: List[int]) -> List[int]:
        """
        Given a list of wiki_ids, returns the page views of each id
        :param wiki_ids: List of wiki ids
        :return:
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
        """
        return self.ranker.page_views(self.page_views_dict, wiki_ids)
