import gzip
import os
import pickle as pkl

import pandas as pd
from google.cloud import storage

from inverted_index_gcp import InvertedIndex

BUCKET_NAME = "ofek_alon_project"


class PickleHandler:
    def __init__(self):
        client = storage.Client()
        self.bucket = client.bucket(bucket_name=BUCKET_NAME)

    def download_from_gcp(self, source_path: str, destination_path: str) -> None:
        blob = self.bucket.get_blob(source_path)
        blob.download_to_filename(destination_path)
        print(f'downloaded {source_path} to {destination_path}')

    def get_index(self, index_name: str):
        source_path = f'postings_gcp_{index_name}/index.pkl'
        destination_path = f'{index_name}.pkl'
        if not self.exists(destination_path):
            self.download_from_gcp(source_path=source_path, destination_path=destination_path)
        return InvertedIndex().read_index(base_dir='.', name=index_name)

    @staticmethod
    def write_pickle_file(path: str, file) -> None:
        with open(path, 'wb') as f:
            pkl.dump(file, f)

    @staticmethod
    def read_pickle_file(path: str):
        with open(path, 'rb') as f:
            file = pkl.loads(f.read())
        return file

    def read_pickle_from_gcp(self, file_name: str):
        if not self.exists(file_name):
            self.download_from_gcp(file_name, file_name)
        return self.read_pickle_file(file_name)

    @staticmethod
    def exists(file_name: str):
        return os.path.exists(file_name)

    def get_page_rank(self):
        file_name = 'page_rank.csv.gz'
        if not self.exists(file_name):
            self.download_from_gcp(file_name, file_name)
        return pd.read_csv(gzip.open(file_name, 'rb'), names=['id', 'page_rank'])
