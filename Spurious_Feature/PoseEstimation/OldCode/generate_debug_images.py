"""
DEBUG VISUALIZATION SCRIPT: ORACLE vs. STANDARD EVALUATION

This script generates images with ground truth & model outputs to identify areas where the models fail in 
pose estimation. It runs models in two distinct modes on the same subset of images:

1. ORACLE MODE (use_gt_boxes=True):
   - Feeds perfect Ground Truth bounding boxes (from COCO) directly to the model.
   - Purpose: Tests the "Theoretical Maximum" of the pose estimator itself.
   - Interpretation: If this fails (Green skeleton exists, Red is missing/wrong), 
     the POSE MODEL is the bottleneck (e.g., cannot handle occlusion/back-facing).

2. STANDARD MODE (use_gt_boxes=False):
   - Uses the model's full pipeline (e.g., YOLO detection + crop).
   - Purpose: Tests "Real World" performance.
   - Interpretation: If this fails but Oracle worked, the DETECTOR is the bottleneck.
   - Interpretation: If red skeletons float in empty space, the DETECTOR is hallucinating.

OUTPUT:
Check the 'debug/' directory for subfolders containing images with Green (GT) vs Red (Prediction) overlays.
"""

import cv2
import numpy as np
# Import your classes (assuming files are in same folder)
from model_evaluation_comparison import PoseEvaluator
from yolov8_pose import YOLOv8PoseEstimator
from movenet import MoveNetEstimator
from mediapipe_pose import MediaPipePoseEstimator
from rtmpose import MMPoseTopDownEstimator

# CONFIGURATION
COCO_ANNOTATIONS = r"EvaluationDatasets\Coco2017\annotations\person_keypoints_val2017.json"
COCO_IMAGES = r"EvaluationDatasets\Coco2017\val2017"
NUM_IMAGES = 20

def main():
    # ... setup evaluator ...
    evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)
    
    # 1. MediaPipe (Oracle Mode)
    mp_estimator = MediaPipePoseEstimator(complexity=2)
    evaluator.visualize_predictions(
        mp_estimator, 
        output_dir='debug/vis_mediapipe_complexity2_oracle', 
        use_gt_boxes=True,
        num_images=NUM_IMAGES
    )

    evaluator.visualize_predictions(
        mp_estimator, 
        output_dir='debug/vis_mediapipe_complexity2_standard', 
        use_gt_boxes=False,
        num_images=NUM_IMAGES
    )

    # 2. MoveNet (Lightning)
    # This will use the GT boxes to crop
    movenet_lightning_estimator = MoveNetEstimator(variant='lightning')
    evaluator.visualize_predictions(
        movenet_lightning_estimator, 
        output_dir='debug/vis_movenet_lightning_oracle', 
        use_gt_boxes=True,
        num_images=NUM_IMAGES
    )

    evaluator.visualize_predictions(
        movenet_lightning_estimator, 
        output_dir='debug/vis_movenet_lightning_standard', 
        use_gt_boxes=False,
        num_images=NUM_IMAGES
    )

    # 3. MoveNet (Thunder)
    movenet_thunder_estimator = MoveNetEstimator(variant='thunder')
    evaluator.visualize_predictions(
        movenet_thunder_estimator, 
        output_dir='debug/vis_movenet_thunder_oracle', 
        use_gt_boxes=True,
        num_images=NUM_IMAGES
    )

    evaluator.visualize_predictions(
        movenet_thunder_estimator, 
        output_dir='debug/vis_movenet_thunder_standard', 
        use_gt_boxes=False,
        num_images=NUM_IMAGES
    )

    # 3. YOLOv8 (Standard Mode)
    yolo_estimator = YOLOv8PoseEstimator(model_size='m')
    evaluator.visualize_predictions(
        yolo_estimator, 
        output_dir='debug/vis_yolov8m_standard', 
        use_gt_boxes=False,
        num_images=NUM_IMAGES
    )

    # 4. MMPose RTM Pose (Standard Mode)
    mmpose_estimator = MMPoseTopDownEstimator(model_variant="w48_256", use_yolo=True)
    evaluator.visualize_predictions(
        mmpose_estimator, 
        output_dir='debug/vis_mmpose_rtmpose_tiny_standard', 
        use_gt_boxes=False,
        num_images=NUM_IMAGES
    )

if __name__ == "__main__":
    main()

# .venv_test\Scripts\python.exe generate_debug_images.py