import numpy as np


class Dataset:
    def __init__(self):
        self._DATA_KEY = 'pixel_data'
        self._LABEL_KEY = 'label'
        self._data = None
        self._n = 0

    def load(self, fn):
        try:
            self._load(fn)
        except RuntimeError as err:
            raise err

    def _load(self, fn):
        raise NotImplementedError

    def size(self):
        return self._n

    def extract(self, indices):
        if not isinstance(indices, int) and not isinstance(indices, list) and not isinstance(indices, np.ndarray):
            raise RuntimeError('invalid type of indices (should be int or list of int)')
