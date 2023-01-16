import pickle
from collections import defaultdict, Counter
from typing import List, Tuple

import numpy as np
from google.cloud import storage

from backend.inverted_index_gcp import InvertedIndex
from tokenizer import Tokenizer
from stemmer import Stemmer
import hashlib

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS


class PreProcessor:
    def __init__(self):
        self.client = storage.Client()
        self.stemmer = Stemmer()
        self.tokenizer = Tokenizer()

    def read_gcp_files(self, paths: List[str], index_type: str):
        parquet_file = self.spark.read.parquet(*paths)
        num_of_docs = parquet_file.count()
        return parquet_file.select(index_type, "id").rdd, num_of_docs

    @staticmethod
    def _reduce_word_counts(word_counts: List[Tuple]) -> List[Tuple]:
        ''' Returns a sorted posting list by wiki_id.
        Parameters:
        -----------
          unsorted_pl: list of tuples
            A list of (wiki_id, tf) tuples
        Returns:
        --------
          list of tuples
            A sorted posting list.
        '''
        return sorted(word_counts)

    @staticmethod
    def calculate_df(postings, as_dict: bool = True):
        ''' Takes a posting list RDD and calculate the df for each token.
        Parameters:
        -----------
          postings: RDD
            An RDD where each element is a (token, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each element is a (token, df) pair.
        '''
        w2df = postings.mapValues(len)
        if as_dict:
            return w2df.collectAsMap()
        return w2df

    @staticmethod
    def calculate_idf(postings, num_of_docs: int, as_dict: bool = True):
        w2idf = postings.mapValues(lambda term: np.log(num_of_docs / len(term)))
        if as_dict:
            return w2idf.collectAsMap()
        return w2idf

    @staticmethod
    def calculate_term_total(postings, as_dict: bool = True):
        term_total = postings.mapValues(lambda term: np.sum([x[1] for x in term]))
        if as_dict:
            return term_total.collectAsMap()
        return term_total

    def calculate_dl(self, doc_text_pairs, index_type: str, as_dict: bool = True):
        dl = doc_text_pairs.map(lambda x: {x['id']: len(self.tokenizer.tokenize(x[index_type]))})
        if as_dict is True:
            return dl.collectAsMap()
        return dl

    @staticmethod
    def partition_postings_and_write(postings, bucket_name: str, index_name: str):
        ''' A function that partitions the posting lists into buckets, writes out
        all posting lists in a bucket to disk, and returns the posting locations for
        each bucket. Partitioning should be done through the use of `token2bucket`
        above. Writing to disk should use the function  `write_a_posting_list`, a
        static method implemented in inverted_index_colab.py under the InvertedIndex
        class.
        Parameters:
        -----------
          postings: RDD
            An RDD where each item is a (w, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each item is a posting locations dictionary for a bucket. The
            posting locations maintain a list for each word of file locations and
            offsets its posting list was written to. See `write_a_posting_list` for
            more details.
        '''
        rdd = postings.map(lambda x: (token2bucket_id(x[0]), [x])).reduceByKey(lambda x, y: x + y)
        posting_locs_rdd = rdd.map(lambda x: InvertedIndex.write_a_posting_list(x, bucket_name, index_name))
        return posting_locs_rdd

    def _word_count(self, text: str, doc_id: int, stemming: bool) -> List[Tuple]:
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
        tokens = self.tokenizer.re_tokenize(text)
        if stemming:
            counter = Counter([self.stemmer.stem(token) for token in tokens])
        else:
            counter = Counter(tokens)
        return [(token, (doc_id, count)) for (token, count) in counter.items()]

    def get_super_posting_locs(self, bucket_name, index_name) -> defaultdict:
        super_posting_locs = defaultdict(list)
        for blob in self.client.list_blobs(bucket_name, prefix=f'postings_gcp_{index_name}'):
            if not blob.name.endswith("pickle"):
                continue
            with blob.open("rb") as f:
                posting_locs = pickle.load(f)
                for k, v in posting_locs.items():
                    super_posting_locs[k].extend(v)
        return super_posting_locs

    def process(self, doc_text_pairs, index_type: str, stemming: bool = False, anchor: bool = False):
        if anchor:
            doc_text_pairs = doc_text_pairs.map(lambda x: x[0]).flatMap(lambda x: x)
            index_type = 'text'
        word_counts = doc_text_pairs.flatMap(lambda x: self._word_count(text=x[index_type], doc_id=x['id'], stemming=stemming))
        postings = word_counts.groupByKey().mapValues(self._reduce_word_counts)
        if anchor:
            postings = postings.mapValues(lambda x: list(set(x)))
        return postings

