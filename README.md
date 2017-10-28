Usage: > python train.py <path_to_csv>

To run unit test, first extract the data to the same directory (c.f. line 6 in test_dataset2d_dicom.py) and type
> python test_dataset2d_dicom.py

=================================

Part 1:

Q1: How did you verify that you are parsing the contours correctly?

By checking the results visually. In fact, even after doing this we cannot be 100% sure whether the inconsistency (if any) is due to mistakes during the manual labeling or bugs in the program.

Q2: What changes did you make to the code, if any, in order to integrate it into our production code base?

Minor change to handle errors more consistently.

Part 2:

Q1: Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

The short answer is no. Will answer possible future modifications with Q3.

Q2: How do you/did you verify that the pipeline was working correctly?

A simple unit test.

Q3: Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?
- In case the dataset is too large to fit in the memory, we will need to use database (e.g. LMDB) and avoid loading all data at once.

- The extraction should run in a different thread suppose the training is done on GPUs; this way the CPU can keep preparing the next batch while the GPUs work on the current batch. However, in my experience, this only matters if the data is not already in the main memory (longer IO time while fetching from a database) or there are computationally expensive data augmentations.

- The extraction should handle cases where images have different sizes (or is different from the target patch size i.e. 256x256). We will need random cropping or padding/mirroring near the boundaries in order to make the patches the same size. In fact, this procedure can be much more complicated if we take into account of the class imbalance problem--for example, we may want comparable numbers of pixels for the majority and minority classes.

=================================

Part 1:

Q1: Discuss any changes that you made to the pipeline you built in Phase 1, and why you made those changes.

Switch the type of labels from boolean to unsigned char to store class labels (1 for heart muscle, 2 for blood pool). Also, We need to load only those slices with both i-contour and o-contour.

Part 2:

Q1: Could you use a simple thresholding scheme to automatically create the i-contours, given the o-contours? Why or why not? Show figures that help justify your answer.

To some extent, yes. However, there will always be some errors (mostly located near the boundaries). Some of the errors can be fixed using morphological operations and hole filling. Will elaborate more with experiment results.

Q2: Do you think that any other heuristic (non-machine learning)-based approaches, besides simple thresholding, would work in this case? Explain.

Assuming that the heart muscle always surround the blood pool, we could generate initial markers for the two regions and then use watershed segmentation to fill in the gap in between the initial markers. Will elaborate more with experiment results.

Experiment results:

The o-contours of SCD0000501 are wrong (about 10 pixels off along the y-axis); I excluded that image from the following evaluation.

Here we test several models (also, with and without morphological closing and hole filling):

a. thresholding (threshold = 0 / Otsu / Li)

We can easily see how thresholding can be useful by looking at the intensity histograms. After normalizing the image to zero mean unit variance, we could then generate a binary image by picking an appropriate threshold value.
Besides using zero as threshold, we could also use other heuristics (i.e. Otsu and Li) to determine the threshold value.

b. thresholding (Gaussian mixture model, GMM)

Assuming that the intensity values of the two classes both follow a Gaussian distribution, we could also determine the threshold value after fitting a Gaussian model with the intensity values.

c. Watershed segmentation

Generate markers then apply standard watershed algorithm.

We evaluate performance using precision, recall, and Jaccard index. The left and right numbers correspond to the result of heart muscle and blood pool.
Results of Models marked with "[postprocessed]" are postprocessed by morphological closing and hole filling.

| Model         | Precision      | Recall        | Jaccard       |
| ------------- | ------------- | ------------- | ------------- |
| threshold (t = 0)  | 0.951 / 0.795  | 0.725 / 0.966  | 0.699 / 0.774        |
| threshold (t = 0), [postprocessed]  | 0.949 / 0.820  | 0.749 / 0.966  | 0.720 / 0.797        |
| threshold (Otsu)  | 0.941 / 0.811  | 0.739 / 0.960  | 0.706 / 0.784        |
| threshold (Otsu), [postprocessed]  | 0.937 / 0.831  | 0.759 / 0.958  | 0.722 / 0.802        |
| threshold (Li)  | 0.915 / 0.848  | 0.774 / 0.946  | 0.722 / 0.809        |
| threshold (Li), [postprocessed]  | 0.909 / 0.867  | 0.796 / 0.944  | 0.737 / 0.825        |
| threshold (GMM)  | 0.911 / 0.819  | 0.741 / 0.942  | 0.691 / 0.780        |
| threshold (GMM), [postprocessed]  | 0.906 / 0.835  | 0.757 / 0.939  | 0.702 / 0.792        |
| Watershed  | 0.848 / 0.931  | 0.874 / 0.915  | 0.756 / 0.857        |
| Watershed, [postprocessed]  | 0.843 / 0.932  | 0.875 / 0.912  | 0.752 / 0.855        |