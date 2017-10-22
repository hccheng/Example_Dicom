from dataset2d_dicom import Dataset2D_DICOM
import numpy as np
import sys

BATCH_SIZE = 8
EPOCHS = 1
SEED = 1

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('usage: python train.py <path_to_csv>')
        exit()

    np.random.seed(SEED)
    try:
        dataset = Dataset2D_DICOM()
        dataset.load(sys.argv[1])
        # print(dataset.size())
        # print(dataset.get_patch(10))

        # iterations / epochs
        for i in range(EPOCHS):
            # shuffle the training samples
            # TODO we could also form batches by sampling with replacement (which is faster)
            shuffle = np.random.permutation(list(range(dataset.size())))

            j = 0
            # iterate all slices
            while j < dataset.size():
                # TODO what should we do for the last, partially-filled batch?
                if j + BATCH_SIZE > dataset.size():
                    break

                # create a batch
                batch = dataset.extract(shuffle[j:j + BATCH_SIZE])
                j += BATCH_SIZE

                # train one iteration with batch
                # batch[0] contains images; batch[1] contains labels
                # ... fill here ...
    except IndexError as e:
        print(e.message)
    except RuntimeError as e:
        print(e.message)
