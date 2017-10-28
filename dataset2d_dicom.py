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
        OCONT_DIR_FORMAT = '{}/contourfiles/{}/o-contours'  # fill in dir, patient id

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

                    icont_dir = ICONT_DIR_FORMAT.format(data_dir, original_id)
                    ocont_dir = OCONT_DIR_FORMAT.format(data_dir, original_id)

                    # load only the intersection of the i- and o-contours
                    try:
                        icont_files = os.listdir(icont_dir)
                        ocont_files = os.listdir(ocont_dir)
                        icont_z = [int(cont_file_pat.match(f).groups()[0]) for f in icont_files]
                        ocont_z = [int(cont_file_pat.match(f).groups()[0]) for f in ocont_files]
                        icont_z, icont_files = zip(*sorted(zip(icont_z, icont_files)))
                        ocont_z, ocont_files = zip(*sorted(zip(ocont_z, ocont_files)))
                        zs = np.intersect1d(icont_z, ocont_z)
                        icont_files = np.array(icont_files)[np.in1d(icont_z, zs)]
                        ocont_files = np.array(ocont_files)[np.in1d(ocont_z, zs)]
                    except:
                        raise RuntimeError('failed to parse the file name of icontours')

                    # iterate all relevant files
                    for z, icont_file, ocont_file in zip(zs, icont_files, ocont_files):
                        # read icontour
                        icont_file_path = os.path.join(icont_dir, icont_file)
                        try:
                            i_mask = poly_to_mask(parse_contour_file(icont_file_path), w, h)
                        except:
                            raise RuntimeError('cannot parse contour file "{}"'.format(icont_file_path))

                        # read ocontour
                        ocont_file_path = os.path.join(ocont_dir, ocont_file)
                        try:
                            o_mask = poly_to_mask(parse_contour_file(ocont_file_path), w, h)
                        except:
                            raise RuntimeError('cannot parse contour file "{}"'.format(ocont_file_path))

                        # read dicom
                        dicom_fn = DICOM_FN_FORMAT.format(data_dir, patient_id, z)
                        try:
                            img = parse_dicom_file(dicom_fn)
                            pixels = img[PIXEL_DATA]
                        except:
                            raise RuntimeError('cannot parse dicom file "{}"'.format(dicom_fn))

                        print(z, ocont_file_path)

                        # make sure everything looks good
                        assert ((h, w) == pixels.shape)
                        assert (i_mask.dtype == np.bool)
                        assert (o_mask.dtype == np.bool)

                        self._data[self._DATA_KEY].append(pixels)
                        # store as uint8 to accommodate for multi-class problems
                        label = np.zeros(o_mask.shape, np.uint8)
                        label[o_mask] = self.LABEL_HEART_MUSCLE
                        label[i_mask] = self.LABEL_BLOOD_POOL
                        self._data[self._LABEL_KEY].append(label)
                        # self._data[self._LABEL_KEY].append(i_mask)

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
