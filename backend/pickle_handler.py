import pickle as pkl


class PickleHandler:
    def __init__(self):
        pass

    @staticmethod
    def write_pickle_file(path: str, file):
        with open(path, 'wb') as f:
            pkl.dump(file, f)

    @staticmethod
    def read_pickle_file(path: str):
        with open(path, 'rb') as f:
            file = pkl.loads(f.read())
        return file