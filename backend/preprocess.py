import pyspark
from google.cloud import storage
from pyspark.shell import spark

from backend.inverted_index import InvertedIndex
from backend.tokenizer import Tokenizer
import hashlib

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS


class PreProcess:
    def __init__(self, bucket_name: str, index_type: str, stemming: bool = False):
        self.bucket_name = bucket_name
        self.index_type = index_type
        self.client = storage.Client()
        self.stemming = stemming

    def _read_gcp_files(self) -> pyspark.RDD:
        full_path = f"gs://{self.bucket_name}/"
        blobs = self.client.list_blobs(self.bucket_name)
        paths = []
        for b in blobs:
            if b.name != 'graphframes.sh':
                self.paths.append(full_path+b.name)
        parquet_file = spark.read.parquet(*paths)
        return parquet_file.select(self.index_type, "id").rdd

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
    def _calculate_df(postings, as_dict: bool = False):
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


    def _partition_postings_and_write(self, postings):
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
        posting_locs_rdd = rdd.map(lambda x: InvertedIndex.write_a_posting_list(x, self.bucket_name))
        return posting_locs_rdd

    def load_and_process(self):
        doc_text_pairs = self._read_gcp_files()
        tokenizer = Tokenizer()
        word_counts = doc_text_pairs.flatMap(lambda x: tokenizer.tokenize(text=x[0], doc_id=x[1], stemming=self.stemming))
        return word_counts.groupByKey().mapValues(self._reduce_word_counts)

