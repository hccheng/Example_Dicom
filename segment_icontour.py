from dataset2d_dicom import Dataset2D_DICOM
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import mixture
from quality_measure import confusion_matrix, calc_measure, print_report
import SimpleITK as sitk
from skimage import filters
from scipy.ndimage.morphology import binary_erosion

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

        j = 0
        cmat = None
        while j < dataset.size():
            d = dataset.extract(j)

            image = d[0]
            label = d[1]

            # normalization
            roi = label > 0
            mean = np.mean(image[roi])
            std = np.std(image[roi])
            norm_image = (image - mean) / std

            # plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
            # plt.subplot(2, 2, 1)
            # plt.imshow(image, cmap='gray')
            # plt.imshow(label, cmap='jet', alpha=0.5)

            # option 1: simple thresholding
            threshold = 0.0
            # threshold = filters.threshold_otsu(norm_image[roi])
            # threshold = filters.threshold_li(norm_image[roi])
            label_predict = np.zeros(label.shape)
            label_predict[roi] = dataset.LABEL_HEART_MUSCLE
            label_predict[np.logical_and(roi, norm_image > threshold)] = dataset.LABEL_BLOOD_POOL

            # option 2: derive cutoff by GMM (not quite working)
            '''
            model = mixture.GaussianMixture(n_components=2, covariance_type='full')
            model.fit([[p] for p in norm_image[roi]])
            pred = model.predict([[p] for p in norm_image.flat]).reshape(norm_image.shape)
            which = np.where(model.means_.flat == np.max(model.means_.flat))[0][0]
            label_predict = np.zeros(label.shape)
            label_predict[roi] = dataset.LABEL_HEART_MUSCLE
            label_predict[np.logical_and(roi, pred == which)] = dataset.LABEL_BLOOD_POOL
            '''

            # option 3: watershed segmentation
            '''
            m1 = binary_erosion(roi, iterations=3) != roi
            m2 = binary_erosion(roi, iterations=6)
            marker = np.zeros(label.shape)
            marker[m1] = dataset.LABEL_HEART_MUSCLE
            marker[m2] = dataset.LABEL_BLOOD_POOL
            result = sitk.MorphologicalWatershedFromMarkers(sitk.GetImageFromArray(image),
                                                            sitk.GetImageFromArray(marker.astype(np.uint8)))
            result = sitk.GetArrayFromImage(result)
            label_predict = np.zeros(label.shape)
            label_predict[roi] = dataset.LABEL_HEART_MUSCLE
            label_predict[np.logical_and(roi, result == dataset.LABEL_BLOOD_POOL)] = dataset.LABEL_BLOOD_POOL

            # plt.imshow(image, cmap='gray')
            # plt.imshow(label_predict, cmap='jet', alpha=0.5)
            '''

            # bins = np.linspace(-2.0, 2.0, 100)
            # plt.subplot(2, 2, 2)
            # plt.hist(norm_image[label > 0], alpha=0.5, histtype='stepfilled', bins=bins)
            # plt.hist(norm_image[label == dataset.LABEL_HEART_MUSCLE], alpha=0.5, histtype='stepfilled',
            #          bins=bins, label='heart muscle')
            # plt.hist(norm_image[label == dataset.LABEL_BLOOD_POOL], alpha=0.5, histtype='stepfilled',
            #          bins=bins, label='blood pool')
            # plt.legend(loc='upper right')

            # plt.subplot(2, 2, 3)
            # plt.imshow(image, cmap='gray')
            # plt.imshow(label_predict, cmap='jet', alpha=0.5)

            # postprocess the binary mask (closing + fill hole)
            # '''
            binary_mask = sitk.GetImageFromArray((label_predict == dataset.LABEL_BLOOD_POOL).astype(np.uint8))
            filled = sitk.BinaryMorphologicalClosing(binary_mask)
            filled = sitk.BinaryFillhole(filled)
            filled = sitk.GetArrayFromImage(filled).astype(np.bool)
            label_predict[filled] = dataset.LABEL_BLOOD_POOL
            # '''

            # plt.subplot(2, 2, 4)
            # plt.imshow(image, cmap='gray')
            # plt.imshow(label_predict, cmap='jet', alpha=0.5)

            # plt.show()
            # exit()

            # accumulate confusion matarix
            if cmat is not None:
                cmat += confusion_matrix(label_predict.astype(np.uint8), label, n_class=3)
            else:
                cmat = confusion_matrix(label_predict.astype(np.uint8), label, n_class=3)

            # next
            j += 1

        # print performance metrics
        print_report(calc_measure(cmat))

    except IndexError as e:
        print(e.message)
    except RuntimeError as e:
        print(e.message)
