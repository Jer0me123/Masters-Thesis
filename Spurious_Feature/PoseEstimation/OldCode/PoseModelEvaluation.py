"""
MULTI-PERSON POSE ESTIMATION EVALUATION SUITE
------------------------------------------------------------------------------
OVERVIEW:
This script provides a standardized framework to evaluate and compare different 
Pose Estimation models (YOLOv8, MoveNet, MediaPipe) on the COCO Keypoints Dataset.

CORE FUNCTIONALITY:
1. Data Loading: 
   - Loads COCO validation images and Ground Truth (GT) annotations.
   - Filters for images containing people.

2. Model Evaluation (evaluate_model):
   - Runs inference on a subset of images.
   - Calculates standard COCO Metrics (Average Precision - AP, Recall - AR).
   - Tracks performance metrics: FPS (Speed) and Detection Rate (% of people found).

3. Oracle vs. Standard Mode ('use_gt_boxes' flag):
   - STANDARD MODE (False): Tests the full pipeline (Detection + Pose). 
     Simulates real-world usage where the model must find people itself.
   - ORACLE MODE (True): Feeds perfect GT bounding boxes to the model.
     Isolates the Pose Estimator's performance by removing detection errors. 
     (Crucial for single-person models like MediaPipe/MoveNet that rely on cropping).

4. visualization (visualize_predictions):
   - Generates side-by-side debug images.
   - Draws Ground Truth skeletons in GREEN.
   - Draws Model Predictions in RED.
   - Helps visually diagnose failures (e.g., missed detections vs. wrong keypoints).

OUTPUT:
- 'evaluation_results.json': Detailed metrics for every model tested.
- 'model_comparison.png': Bar charts comparing Accuracy (AP), Speed (FPS), and Detection Rate.
- Console Summary: A clean text table ranking the models.

WHY 'use_gt_boxes' (ORACLE MODE)?

Many pose estimation models (like MediaPipe and MoveNet) are designed as "Single Person" models.
To work on the COCO dataset (which has multiple people per image), they typically require a 
two-stage process:
    Stage 1: A detector (like YOLO) finds the bounding box of every person.
    Stage 2: The pose model crops that box and estimates keypoints inside it.

If the Stage 1 detector fails (misses a person or draws a bad box), the Stage 2 pose model 
will fail, even if the pose model itself is perfect.

The 'use_gt_boxes' flag allows us to isolate these errors:

1. Standard Mode (use_gt_boxes=False):
   - Uses the full pipeline (YOLO Detector -> Pose Model).
   - Measures "Real World" performance. 
   - Low score could mean BAD DETECTOR or BAD POSE MODEL.

2. Oracle Mode (use_gt_boxes=True):
   - Feeds perfect Ground Truth bounding boxes from COCO directly to the pose model.
   - Bypasses the detector entirely.
   - Measures "Theoretical Maximum" performance of the pose model.
   - If this score is high but Standard Mode is low, your DETECTOR is the bottleneck.
------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

class PoseEvaluator:
    """
    Evaluate pose estimation models on COCO validation set
    """
    def __init__(self, coco_path, images_path):
        """
        coco_path: path to COCO annotations JSON (person_keypoints_val2017.json)
        images_path: path to COCO val2017 images folder
        """
        self.coco_path = coco_path
        self.images_path = images_path
        self.coco_gt = COCO(coco_path)
        
        # Get all images with person annotations
        cat_ids = self.coco_gt.getCatIds(catNms=['person'])
        self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)

    def evaluate_model(self, model, num_images=500, model_name='Model', use_gt_boxes=False):
        results = []
        person_cat_id = self.coco_gt.getCatIds(catNms=['person'])[0]
        
        # Select the specific images we are going to test
        subset_img_ids = self.img_ids[:num_images]
        
        print(f"Evaluating {model_name} on {len(subset_img_ids)} images...")
        if use_gt_boxes:
            print("(!) ATTEMPTING TO USE GROUND TRUTH BOXES (Oracle Mode)")
            
        start_time = time.time()
        
        for i, img_id in enumerate(tqdm(subset_img_ids)):
            # Load Image
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_path, img_info['file_name'])
            image = cv2.imread(img_path)
            
            if image is None: continue
            
            # --- 1. FETCH GROUND TRUTH IF ENABLED ---
            current_gt_boxes = None
            if use_gt_boxes:
                ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False)
                anns = self.coco_gt.loadAnns(ann_ids)
                current_gt_boxes = [ann['bbox'] for ann in anns]
                
                # If in Oracle mode and no people exist, skip processing
                if not current_gt_boxes:
                    continue

            # --- 2. PREDICT WITH COMPATIBILITY CHECK ---
            try:
                # Attempt to pass gt_boxes. 
                # If model is 'Old' (doesn't accept gt_boxes), this raises TypeError.
                if use_gt_boxes and current_gt_boxes is not None:
                    predictions = model.predict(image, gt_boxes=current_gt_boxes)
                else:
                    predictions = model.predict(image)
            except TypeError:
                # Fallback for Old Models: Run standard prediction ignoring gt_boxes
                predictions = model.predict(image)

            # Safety check for old models that might return None
            if predictions is None:
                predictions = []
            
            # --- 3. PROCESS RESULTS ---
            for pred in predictions:
                keypoints = pred['keypoints']
                score = pred['score']
                keypoints_flat = keypoints.flatten().tolist()
                
                # Use detected bbox if available, otherwise 0s (irrelevant for Keypoint AP)
                bbox = pred.get('bbox', [0,0,0,0]).tolist() if isinstance(pred.get('bbox'), np.ndarray) else [0,0,0,0]

                results.append({
                    'image_id': img_id,
                    'category_id': person_cat_id,
                    'keypoints': keypoints_flat,
                    'score': score,
                    'bbox': bbox 
                })
        
        total_time = time.time() - start_time
        
        # --- PERFORM COCO EVALUATION ---
        if not results:
            print(f"Warning: {model_name} made 0 detections.")
            return {
                'model': model_name, 'AP': 0.0, 'AP_50': 0.0, 'AP_75': 0.0, 'AR': 0.0,
                'fps': 0.0, 'avg_time_ms': 0.0, 'total_detections': 0, 'total_gt': 0, 'detection_rate': 0.0
            }

        # 1. Load results into COCO object
        coco_dt = self.coco_gt.loadRes(results)
        
        # 2. Run Evaluation
        # Restrict evaluation to only the subset of images processed
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = subset_img_ids 
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 3. Extract Metrics
        stats = coco_eval.stats # [AP, AP50, AP75, APm, APl, AR, AR50, AR75, ARm, ARl]
        
        # Calculate speed metrics
        total_images = len(subset_img_ids)
        avg_time = (total_time / total_images) * 1000
        fps = 1.0 / (total_time / total_images) if total_time > 0 else 0
        
        # Calculate Detection Rate correctly
        total_gt_people = 0
        for img_id in subset_img_ids:
            ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False)
            anns = self.coco_gt.loadAnns(ann_ids)
            # FILTER: Only count people who actually have keypoints labeled
            valid_anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
            total_gt_people += len(valid_anns)

        return {
            'model': model_name,
            'AP': stats[0], #AP@50-95
            'AP_50': stats[1],
            'AP_75': stats[2],
            'AR': stats[5],
            'fps': fps,
            'avg_time_ms': avg_time,
            'total_detections': len(results),
            'total_gt': total_gt_people,
            'detection_rate': len(results) / total_gt_people if total_gt_people > 0 else 0
        }

    def compare_models(self, models_dict, num_images=500, output_dir="."):
        """
        Compare multiple models
        """
        all_metrics = []
        
        for model_name, model_config in models_dict.items():
            model, use_gt_boxes = model_config
            
            metrics = self.evaluate_model(model, num_images, model_name, use_gt_boxes=use_gt_boxes)
            all_metrics.append(metrics)
        
        # Create comparison visualizations
        self._plot_comparison(all_metrics, output_dir)
        
        # Print summary table
        self._print_summary_table(all_metrics)
        
        return all_metrics
    
    def _print_summary_table(self, metrics_list):
        """Print comparison table"""
        print(f"\n{'='*120}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*120}")
        print(f"{'Model':<25} {'AP':>8} {'AP@50':>8} {'AP@75':>8} {'AR':>8} {'FPS':>8} {'Time(ms)':>10} {'Detections':>12} {'Det Rate':>10}")
        print(f"{'-'*120}")
        
        for m in sorted(metrics_list, key=lambda x: x['AP'], reverse=True):
            det_str = f"{m['total_detections']}/{m['total_gt']}"
            print(f"{m['model']:<25} {m['AP']:>8.3f} {m['AP_50']:>8.3f} {m['AP_75']:>8.3f} "
                  f"{m['AR']:>8.3f} {m['fps']:>8.1f} {m['avg_time_ms']:>10.2f} {det_str:>12} {m['detection_rate']*100:>9.1f}%")
    
    def _plot_comparison(self, metrics_list, output_dir):
        """Create comparison plots"""
        out_path = os.path.join(output_dir, "model_comparison.png")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = [m['model'] for m in metrics_list]
        aps = [m['AP'] for m in metrics_list]
        fps_values = [m['fps'] for m in metrics_list]
        det_rates = [m['detection_rate'] * 100 for m in metrics_list]
        
        # AP comparison
        axes[0].barh(models, aps, color='skyblue')
        axes[0].set_xlabel('Average Precision (AP)')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xlim(0, 1)
        
        # FPS comparison
        axes[1].barh(models, fps_values, color='lightcoral')
        axes[1].set_xlabel('Frames Per Second (FPS)')
        axes[1].set_title('Speed Comparison')
        
        # Detection rate comparison
        axes[2].barh(models, det_rates, color='lightgreen')
        axes[2].set_xlabel('Detection Rate (%)')
        axes[2].set_title('Detection Rate')
        axes[2].set_xlim(0, 120)
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to '{out_path}'")
        plt.show()

    def visualize_predictions(self, model, num_images=20, output_dir='debug_vis', use_gt_boxes=True):
        """
        Runs the model and saves images with GT (Green) vs Prediction (Red) overlay.
        Compatible with ALL models (MediaPipe, MoveNet, YOLOv8).
        """
        import shutil
        
        # 1. Setup Directories
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"(!) Generating {num_images} debug images in '{output_dir}/'...")
        
        # COCO Skeleton topology (0-based index)
        SKELETON = [
            (15,13), (13,11), (16,14), (14,12), # Legs
            (11,12), (5,11), (6,12),            # Hips & Torso
            (5,6), (5,7), (6,8),                # Shoulders
            (7,9), (8,10),                      # Arms
            (1,2), (0,1), (0,2), (1,3), (2,4)   # Face
        ]
        
        person_cat_id = self.coco_gt.getCatIds(catNms=['person'])[0]
        subset_ids = self.img_ids[:num_images]

        for i, img_id in enumerate(tqdm(subset_ids)):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_path, img_info['file_name'])
            original_img = cv2.imread(img_path)
            
            if original_img is None: continue
            
            vis_img = original_img.copy()
            
            # --- 1. GET GROUND TRUTH DATA ---
            ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False)
            anns = self.coco_gt.loadAnns(ann_ids)
            
            gt_boxes_list = []
            
            for ann in anns:
                # Store box for "Oracle Mode"
                gt_boxes_list.append(ann['bbox']) 
                
                # DRAW GT SKELETON (GREEN)
                # Skip drawing if no keypoints (num_keypoints=0)
                if ann.get('num_keypoints', 0) == 0: continue
                
                kp = ann['keypoints']
                for p1, p2 in SKELETON:
                    x1, y1, v1 = kp[p1*3], kp[p1*3+1], kp[p1*3+2]
                    x2, y2, v2 = kp[p2*3], kp[p2*3+1], kp[p2*3+2]
                    if v1 > 0 and v2 > 0:
                        cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # --- 2. RUN PREDICTION (ROBUST MODE) ---
            preds = []
            try:
                # Try passing GT boxes (Works for MP, MoveNet)
                if use_gt_boxes and gt_boxes_list:
                    preds = model.predict(original_img, gt_boxes=gt_boxes_list)
                else:
                    preds = model.predict(original_img)
            except TypeError:
                # Fallback for models that don't accept gt_boxes (YOLOv8)
                preds = model.predict(original_img)
            
            # Safety check
            if preds is None: preds = []

            # --- 3. DRAW PREDICTIONS (RED) ---
            for p in preds:
                kps = p['keypoints'] # Numpy [17, 3]
                
                for p1, p2 in SKELETON:
                    x1, y1, conf1 = kps[p1]
                    x2, y2, conf2 = kps[p2]
                    
                    # Draw even low confidence lines to debug "Hallucinations"
                    if conf1 > 0.05 and conf2 > 0.05:
                        cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Save
            save_path = os.path.join(output_dir, f"debug_{img_id}.jpg")
            cv2.imwrite(save_path, vis_img)
            
        print(f"Done! Check the '{output_dir}' folder.")


# Example usage
if __name__ == "__main__":
    # Setup paths
    COCO_ANNOTATIONS = r"EvaluationDatasets\Coco2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"EvaluationDatasets\Coco2017\val2017"
    
    parser = argparse.ArgumentParser(description="Multi-person pose evaluation on COCO")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where evaluation results and plots will be saved"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)
    
    # Import models (make sure to use the updated multi-person version!)
    from yolov8_pose import YOLOv8PoseEstimator
    from movenet import MoveNetEstimator
    from mediapipe_pose import MediaPipePoseEstimator
    from rtmpose import MMPoseTopDownEstimator
    
    # Initialize models to compare
    # Format: 'Name': (ModelInstance, use_gt_boxes)
    models_to_test = {
        'YOLOv8s-Pose': (YOLOv8PoseEstimator(model_size='s'), False),
        # 'YOLOv8m-Pose': (YOLOv8PoseEstimator(model_size='m'), False),
        # 'YOLOv8l-Pose': (YOLOv8PoseEstimator(model_size='l'), False),

        # 'MoveNet-Lightning (Standard)': (MoveNetEstimator(variant='lightning'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector
        # 'MoveNet-Lightning (Oracle)': (MoveNetEstimator(variant='lightning'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector

        # 'MoveNet-Thunder (Standard)': (MoveNetEstimator(variant='thunder'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector
        # 'MoveNet-Thunder (Oracle)': (MoveNetEstimator(variant='thunder'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector
        
        # 'MediaPipe-C2 (Standard)': (MediaPipePoseEstimator(complexity=2), False), # requires an external person detector
        # 'MediaPipe-C2 (Oracle)':   (MediaPipePoseEstimator(complexity=2), True), # requires an external person detector
        
        # "HRNet-W32 (Standard)": (MMPoseTopDownEstimator("w32_256", use_yolo=True), False), # requires an external person detector
        # "HRNet-W32 (Oracle)": (MMPoseTopDownEstimator("w32_256", use_yolo=True), True), # requires an external person detector

        # "HRNet-W32-384 (Standard - Large)": (MMPoseTopDownEstimator("w32_384", use_yolo=True), False), # requires an external person detector
        # "HRNet-W32-384 (Oracle - Large)": (MMPoseTopDownEstimator("w32_384", use_yolo=True), True), # requires an external person detector

        # "HRNet-W48 (Standard)": (MMPoseTopDownEstimator("w48_256", use_yolo=True), False), # requires an external person detector
        # "HRNet-W48 (Oracle)": (MMPoseTopDownEstimator("w48_256", use_yolo=False), True), # requires an external person detector
    }
    
    # Run comparison
    print("\n" + "="*80)
    print("MULTI-PERSON POSE ESTIMATION EVALUATION")
    print("="*80)
    
    results = evaluator.compare_models(models_to_test, num_images=len(evaluator.img_ids), output_dir=args.output_dir)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {results_path}")

# .venv_test\Scripts\python.exe PoseModelEvaluation.py --output_dir "EvaluationDatasets\Coco2017"