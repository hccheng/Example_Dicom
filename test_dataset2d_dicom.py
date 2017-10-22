from dataset2d_dicom import Dataset2D_DICOM
import unittest
import os
import numpy as np

TESTDATA_CSV = os.path.join(os.path.dirname(__file__), 'final_data/link.csv')


class TestDatasetExtraction(unittest.TestCase):
    def test_split(self):
        dataset = Dataset2D_DICOM()
        dataset.load(TESTDATA_CSV)

        self.assertEqual(dataset.size(), 96)

        # extract one at a time
        for i in range(dataset.size()):
            d = dataset.extract(i)
            self.assertEqual(len(d), 2)
            self.assertEqual(d[0].shape, d[1].shape)
            self.assertEqual(d[0].dtype, np.int16)
            self.assertEqual(d[1].dtype, np.bool)

        # extract a batch at a time
        for i in range(100):
            shuffle = np.random.permutation(list(range(dataset.size())))
            batch = dataset.extract(shuffle[:8])

            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].shape, batch[1].shape)
            self.assertEqual(batch[0].dtype, np.int16)
            self.assertEqual(batch[1].dtype, np.bool)

            for j, d in enumerate(batch[0]):
                self.assertTrue(np.array_equal(d, dataset.extract(shuffle[j])[0]))

            for j, d in enumerate(batch[1]):
                self.assertTrue(np.array_equal(d, dataset.extract(shuffle[j])[1]))

        # if we want to visually check the images/labels
        '''
        from matplotlib import pyplot as plt
        shuffle = np.random.permutation(list(range(dataset.size())))
        batch = dataset.extract(shuffle[:8])
        for i in range(8):
            plt.imshow(batch[0][i], cmap='gray')
            plt.imshow(batch[1][i], alpha=0.5, cmap='Set1_r')
            plt.show()
        '''


if __name__ == '__main__':
    SEED = 5
    np.random.seed(SEED)
    unittest.main()
