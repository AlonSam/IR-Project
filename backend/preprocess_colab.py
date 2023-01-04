from typing import List, Tuple
from inverted_index_colab import InvertedIndex
from tokenizer import Tokenizer
import hashlib
import logging

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS


class PreProcess:
    def __init__(self, index_type: str, index_name: str, stemming: bool = False):
        self.index_type = index_type
        self.index_name = index_name
        self.stemming = stemming

    def read_files(self, path):
        parquet_file = spark.read.parquet(path)
        logging.info('Created parquet file')
        return parquet_file.limit(1000).select(self.index_type, "id").rdd

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
    def calculate_df(postings, as_dict: bool = False):
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

    def partition_postings_and_write(self, postings):
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
        posting_locs_rdd = rdd.map(lambda x: InvertedIndex.write_a_posting_list(x, self.bucket_name, self.index_name))
        return posting_locs_rdd

    def load_and_process(self, path):
        doc_text_pairs = self.read_files(path)
        logging.info('Successfully read files')
        tokenizer = Tokenizer()
        word_counts = doc_text_pairs.flatMap(lambda x: tokenizer.tokenize(text=x[0], doc_id=x[1], stemming=self.stemming))
        postings = word_counts.groupByKey().mapValues(self._reduce_word_counts)
        return postings

