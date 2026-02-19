This text file outlines the purpose of each .ipynb & .py file in this directory. 

Setup.ipynb - This file outlines the Environment setup

<!-- Gender Classification -->
GenderClassificationModels\GenderClassificationTesting.ipynb - This notebook serves as a testing ground for evaluating the performance of various gender classification models:
    1. Hugging Face Gender Classification
    2. Realistic Gender Classifier
    3. DeepFace
    4. InsightFace
    5. FairFace

<!-- SkinTone Classification -->
SkinToneClassificationModels\SkinToneDetectionTesting.ipynb - This notebook serves as a testing ground for evaluating the performance of various skin tone classification models:
    1. SkinTone Classifier Library 
    2. RandomForest
    3. DenseNet 121
    4. VGG16

GenderSkinToneAnnotation.py - This python script performs gender & skintone image annotation using the  Realistic Gender Classifier & VGG16 Trained Models

AnnotationDistributionAssessment.ipynb - This notebook serves to visualise the gender and skin tone label distribution across Re-Laion5B & SD generated images.

FacialDetectionEvaluation.ipynv - This notebook serves as a testing ground for evaluating the performance of various face detector models:
    1. Retina Face
    2. SCRFD
    3. Yolo-Face
    4. Mediapipe Face Detection

RELAion5B_Subset_Image_Retrieval.py - This python script serves to curate a high-quality subset of images from a larger dataset (LAION-5B), based on similarity scores.

DatasetPreperation.ipynb - This notebook outlines how to download the CCv2, FACET & MST-E datasets and processes via face segmentations needed for gender & skin tone model training / testing.

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Download and segments the CCv2, FACET & MST-E datasets -> G:\Thesis\MonkSkinTone_Dataset | G:\Thesis\FACET_Dataset | G:\Thesis\CasualConversationv2_Dataset
2. Train various SkinTone annotaion models
3. Test various gender & skin tone annotation models
4. Annotate datasets using gender & skin tone models -> E:\ImageRetrieval\Professions_125k_ISCO_Aligned_Annotations\annotations.jsonl | E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl