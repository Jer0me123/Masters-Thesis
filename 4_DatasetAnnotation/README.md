# Dataset Annotation Pipeline

## Overview

This directory contains the pipeline for annotating profession-conditioned images with gender and skin tone labels.

The pipeline consists of:

1. Training data preparation (CCv2, MST-E, FACET datasets)
2. Face detector evaluation and selection
3. Gender classification model evaluation and selection
4. Skin tone classification model training and evaluation
5. Gender and skin tone annotation of Re-LAION-5B and SD datasets
6. Annotation distribution assessment

The scripts and notebooks in this directory have been re-tested and verified to be functioning correctly.

## Files

### `DatasetPreperation.ipynb`

Downloads and processes the training datasets required for gender and skin tone model development.

Datasets:
- CasualConversations v2 (CCv2)
- Monk Skin Tone E (MST-E)
- FACET

Functionality:
- Applies MediaPipe FaceMesh segmentation to each dataset
- Converts MST labels to 3-bin groupings (Light/Mid/Dark)

### `FacialDetectionEvaluation.ipynb`

Evaluates face detection models on the WIDER Face validation set to select the best detector for the annotation pipeline.

Models evaluated:
- RetinaFace
- SCRFD
- YOLO-Face
- MediaPipe Face Detection

Conclusion: YOLO-Face selected as the production detector.

### `GenderClassificationModels/GenderClassificationTesting.ipynb`

Evaluates gender classification models on CCv2 and FACET datasets.

Models evaluated:
- HuggingFace Gender Classification
- Realistic Gender Classifier
- DeepFace
- InsightFace
- FairFace

Conclusion: Realistic Gender Classifier selected as the production model.

### `SkinToneClassificationModels/SkinToneDetectionTesting.ipynb`

Evaluates skin tone classification models on CCv2, MST-E, and FACET datasets.

Models evaluated:
- SkinTone Classifier Library
- Random Forest
- DenseNet121
- VGG16

Conclusion: VGG16 (regression, RGB, 10-class) selected as the production model.

### `SkinToneClassificationModels/DenseNet121_SkinTone_Training.py`

Training script for DenseNet121-based MST skin tone classification.

### `SkinToneClassificationModels/RandomForest_SkinTone_Training.py`

Training script for Random Forest MST skin tone classification using histogram-based colour features.

### `SkinToneClassificationModels/VGG16_MST_Testing/vgg16_mst_classification_regression_rgb_lab.py`

Primary training script supporting VGG16 and ResNet18 across multiple configurations.

Supports:
- Architecture: `vgg16` | `resnet18`
- Mode: `classification` | `regression` | `coral`
- Input space: `rgb` | `lab`
- Class counts: 3, 4, 10

### `SkinToneClassificationModels/VGG16_MST_Testing/create_background_corrected_dataset.py`

Replaces black background pixels in segmented face images with the mean face colour. Used to generate `_BGFixed` dataset variants for training.

### `GenderSkinToneAnnotation.py`

Annotates images with gender and skin tone predictions using the selected production models.

- Gender: Realistic Gender Classifier
- Skin tone: VGG16 (configurable via CLI)

### `RELaion5B_Subset_Image_Retrieval.py`

Selects the top-N images per profession from the full Re-LAION-5B retrieval results based on CLIP similarity score encoded in the filename. Copies selected images, facemesh crops, and truncates the annotations JSONL to the selected subset.

### `AnnotationDistributionAssessment.ipynb`

Visualises gender and skin tone label distributions across the Re-LAION-5B and SD datasets for the full retrieval set and the 1k subset.

### `Setup.ipynb`

Documents and performs environment setup required for:
- PyTorch / torchvision
- Ultralytics YOLO
- MediaPipe
- InsightFace / ONNX Runtime
- Skin tone classifier dependencies
- Supporting packages