from dataset import Dataset
from parsing import parse_dicom_file, parse_contour_file, poly_to_mask
import csv
import os
import re
import numpy as np


class Dataset2D_DICOM(Dataset):
    def __init__(self):
        """Initialize a dataset
        """
        Dataset.__init__(self)

    def _load(self, fn):
        """Load the dataset stored in a csv file (implementation)

        :param fn: filepath to the csv file
        """

        if not os.path.isfile(fn):
            raise RuntimeError('file does not exist / input is not a file')
        data_dir = os.path.dirname(fn)

        # regexp for extracting z-index from file name
        cont_file_pat = re.compile('IM-0001-(\d+)-.contour-manual.txt')

        PIXEL_DATA = 'pixel_data'
        PATIENT_ID = 'patient_id'
        ORIGINAL_ID = 'original_id'
        DICOM_FN_FORMAT = '{}/dicoms/{}/{}.dcm'  # fill in dir, patient id, and z-index
        ICONT_DIR_FORMAT = '{}/contourfiles/{}/i-contours'  # fill in dir, patient id

        try:
            with open(fn, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self._data = {self._DATA_KEY: [], self._LABEL_KEY: []}
                for row in reader:
                    patient_id = row[PATIENT_ID]
                    original_id = row[ORIGINAL_ID]

                    # get the dimensions of the first slice in this volume
                    z = 1
                    dicom_fn = DICOM_FN_FORMAT.format(data_dir, patient_id, z)
                    img = parse_dicom_file(dicom_fn)
                    pixels = img[PIXEL_DATA]
                    h, w = pixels.shape

                    cont_dir = ICONT_DIR_FORMAT.format(data_dir, original_id)
                    # iterate all files in the directory
                    for cont_file in os.listdir(cont_dir):
                        # read icontour
                        cont_file_path = os.path.join(cont_dir, cont_file)
                        try:
                            mask = poly_to_mask(parse_contour_file(cont_file_path), w, h)
                        except:
                            raise RuntimeError('cannot parse contour file "{}"'.format(cont_file_path))

                        # extract the z-index
                        try:
                            z = int(cont_file_pat.match(cont_file).groups()[0])
                        except:
                            raise RuntimeError('failed to parse the file name of icontours')

                        # read dicom
                        dicom_fn = DICOM_FN_FORMAT.format(data_dir, patient_id, z)
                        try:
                            img = parse_dicom_file(dicom_fn)
                            pixels = img[PIXEL_DATA]
                        except:
                            raise RuntimeError('cannot parse dicom file "{}"'.format(dicom_fn))

                        # make sure everything looks good
                        assert ((h, w) == pixels.shape)
                        assert (mask.dtype == np.bool)

                        self._data[self._DATA_KEY].append(pixels)
                        # store as uint8 to accommodate for multi-class problems
                        # self._data[self._LABEL_KEY].append(mask.astype(np.uint8))
                        self._data[self._LABEL_KEY].append(mask)

                self._data[self._DATA_KEY] = np.array(self._data[self._DATA_KEY])
                self._data[self._LABEL_KEY] = np.array(self._data[self._LABEL_KEY])
                self._n = len(self._data[self._LABEL_KEY])
        except IOError:
            raise RuntimeError('cannot open csv file "{}"'.format(fn))

    def extract(self, indices):
        """Extract slice(s) from the dataset

        :param indices: an index or a list of indices
        :return: corresponding image(s) and label(s)
        """

        if not isinstance(indices, int) and not isinstance(indices, list) and not isinstance(indices, np.ndarray):
            raise RuntimeError('invalid type of indices (should be int or list of int)')
        return self._data[self._DATA_KEY][indices], self._data[self._LABEL_KEY][indices]
