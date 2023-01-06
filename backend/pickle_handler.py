import pickle as pkl
from google.cloud import storage

from backend.inverted_index_gcp import InvertedIndex

BUCKET_NAME = "ofek_alon_project"


class PickleHandler:
    def __init__(self):
        client = storage.Client()
        self.bucket = client.bucket(bucket_name=BUCKET_NAME)

    def download_from_gcp(self, source_path: str, destination_path: str) -> None:
        blob = self.bucket.get_blob(source_path)
        blob.download_to_filename(destination_path)

    def get_index(self, index_name: str):
        source_path = f'postings_gcp_{index_name}/index.pkl'
        destination_path = f'{index_name}.pkl'
        self.download_from_gcp(source_path=source_path, destination_path=destination_path)
        return InvertedIndex().read_index(base_dir='.', name=destination_path)

    @staticmethod
    def write_pickle_file(path: str, file) -> None:
        with open(path, 'wb') as f:
            pkl.dump(file, f)

    @staticmethod
    def read_pickle_file(path: str):
        with open(path, 'rb') as f:
            file = pkl.loads(f.read())
        return file