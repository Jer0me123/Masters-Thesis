This text file outlines the purpose of each .ipynb & .py file in this directory. 

Setup.ipynb - This file outlines the Environment setup
PoseModels\ - This directory includes the .py file pose models which were tested & used.
PoseModelEvaluation.py - This python script containes a variety of pose models tested on inference speed & output quality to determine which Model is most applicable.
PoseDetection.py - This python script detects poses using Yolov8 within the specified images, producing keypoints and visualising the detected poses.
YoloHyperparamTuning.py - This python script serves as a Hyperparameter Tuning Suite for YOLOv8 Pose Estimation.

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Poses visualised on input images & poses.jsonl files of the specified directories -> F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\PoseDetection |F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\PoseDetection (Coco is not used here since this requires images to 100% contain people)