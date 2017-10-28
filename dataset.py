class Dataset:
    def __init__(self):
        """Initialize a dataset
        """
        self._DATA_KEY = 'pixel_data'
        self._LABEL_KEY = 'label'
        self.LABEL_HEART_MUSCLE = 1  # heart muscle = 1
        self.LABEL_BLOOD_POOL = 2  # blood pool = 2
        self._data = None
        self._n = 0

    def load(self, fn):
        """Load the dataset stored in a csv file

        :param fn: filepath to the csv file
        """

        try:
            self._load(fn)
        except RuntimeError as err:
            raise err

    def _load(self, fn):
        """Load the dataset stored in a csv file (implementation)

        :param fn: filepath to the csv file
        """

        raise NotImplementedError

    def size(self):
        """Get the size of the dataset

        :return: total number of slices in the dataset
        """

        return self._n

    def extract(self, indices):
        """Extract slice(s) from the dataset

        :param indices: an index or a list of indices
        :return: corresponding image(s) and label(s)
        """

        raise NotImplementedError
