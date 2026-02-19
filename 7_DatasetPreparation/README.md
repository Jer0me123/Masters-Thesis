This text file outlines the purpose of each .ipynb & .py file in this directory. 

1_CreateSplits.py -> This scripts create the base train/test/validation splits used for model training: UniversalSplits\Professions_125k_ISCO_Aligned_1k_Subset | UniversalSplits\StableDiffusion
2_CreateObject_OHE_splits.py -> This script converts object detection outputs into fixed-length feature vectors i.e., vector of 1/0s based on if an object apppears in the image or not.
3_Create_Poses_MeanRGB_OHE_Splits.py -> This script adds auxiliary features (Pose keypoints, Mean RGB stats, Depth features, Object encodings) to existing splits.
4_CreateCaption_Splits.py -> This script injects captions into dataset splits.
5_ExtendImagePathsWithSuffix.py -> This script updates the image paths in the .json files to refer to the correct images via the suffix. 
6_MergeSplitsForDatasetClassification.py -> This script creates binary dataset-classification splits, to allow for dataset classification.
7_PseudoDatasetSplitsForDatasetClassification.py -> This script uses existing image splits but assigns arbitrary labels to image, aimed to serve as a negative control experiment.
8_ExtendWithCoco.py -> This script extend the result of 6_MergeSplitsForDatasetClassification.py by adding the Coco dataset image refernces to cahnge the datase classification from a binary to a non-binary problem.

UniversalSplits\ -> This directory contains the splits used to train the dataset / gender / skintone spurious feature classifiers.

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Generates .json files used for training the spuriosu feature identification models