import cv2
import numpy as np
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class PoseEvaluator:
    """
    Evaluate pose estimation models on COCO validation set
    NOW SUPPORTS MULTI-PERSON EVALUATION!
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
    
    # def evaluate_model(self, model, num_images=500, model_name='Model',  use_gt_boxes=False):
    #     """
    #     Evaluate model on ALL people in images (proper COCO evaluation)
        
    #     Model should have predict(image) method that returns:
    #     List[Dict] with each dict containing:
    #         - 'keypoints': np.array([17, 3]) 
    #         - 'score': float
    #         - 'bbox': np.array([4]) in xyxy format
    #     """
    #     print(f"\n{'='*60}")
    #     print(f"Evaluating {model_name} (Multi-Person)")
    #     print(f"{'='*60}")

    #     test_img_ids = self.img_ids[:num_images]

    #     results = []
    #     inference_times = []
    #     total_detections = 0
    #     total_gt_people = 0

    #     for img_id in tqdm(test_img_ids, desc=f"Processing {model_name}"):
    #         img_info = self.coco_gt.loadImgs(img_id)[0]
    #         img_path = os.path.join(self.images_path, img_info['file_name'])

    #         image = cv2.imread(img_path)
    #         if image is None:
    #             continue

    #         # Get all person annotations with keypoints
    #         ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
    #         anns = self.coco_gt.loadAnns(ann_ids)
    #         anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
            
    #         if len(anns) == 0:
    #             continue
            
    #         total_gt_people += len(anns)

    #         # Run inference - get ALL people
    #         start = time.time()
    #         predictions = model.predict(image)  # Returns list of detections
    #         inference_times.append(time.time() - start)
            
    #         # Handle case where predict returns None or empty list
    #         if predictions is None:
    #             predictions = []
            
    #         total_detections += len(predictions)

    #         # Process each detected person
    #         for pred in predictions:
    #             # Convert keypoint confidences to COCO visibility format
    #             # COCO format: v=0 (not labeled), v=1 (labeled but not visible), v=2 (labeled and visible)
    #             kp = pred['keypoints'].copy()
    #             conf = kp[:, 2]
                
    #             # Set visibility based on confidence threshold
    #             kp[:, 2] = np.where(conf > 0.5, 2, np.where(conf > 0.2, 1, 0))
    #             keypoints = kp.flatten().tolist()

    #             # Convert bbox to xywh format for COCO
    #             x1, y1, x2, y2 = pred['bbox']
    #             bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    #             results.append({
    #                 'image_id': int(img_id),
    #                 'category_id': 1,
    #                 'keypoints': [float(x) for x in keypoints],
    #                 'bbox': bbox,
    #                 'score': float(pred['score'])
    #             })

    #     # Calculate statistics
    #     avg_time = np.mean(inference_times) * 1000 if inference_times else 0
    #     fps = 1000.0 / avg_time if avg_time > 0 else 0

    #     print(f"\nDetection Statistics:")
    #     print(f"  Total GT people: {total_gt_people}")
    #     print(f"  Total detections: {total_detections}")
    #     print(f"  Detection rate: {100*total_detections/total_gt_people:.1f}%")
    #     print(f"  Avg detections per image: {total_detections/len(test_img_ids):.1f}")

    #     print(f"\nSpeed Metrics:")
    #     print(f"  Average inference time: {avg_time:.2f} ms")
    #     print(f"  FPS: {fps:.2f}")

    #     # Save results
    #     results_file = f'{model_name.replace(" ", "_")}_results.json'
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f)

    #     # Run COCO evaluation
    #     if len(results) > 0:
    #         print(f"\nAccuracy Metrics (COCO AP):")
    #         coco_dt = self.coco_gt.loadRes(results_file)
    #         coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
    #         coco_eval.params.imgIds = test_img_ids
    #         coco_eval.params.useCats = 1
    #         coco_eval.params.catIds = [1]

    #         coco_eval.evaluate()
    #         coco_eval.accumulate()
    #         coco_eval.summarize()

    #         metrics = {
    #             'model': model_name,
    #             'AP': coco_eval.stats[0],
    #             'AP_50': coco_eval.stats[1],
    #             'AP_75': coco_eval.stats[2],
    #             'AP_medium': coco_eval.stats[3],
    #             'AP_large': coco_eval.stats[4],
    #             'AR': coco_eval.stats[5],
    #             'avg_time_ms': avg_time,
    #             'fps': fps,
    #             'total_detections': total_detections,
    #             'total_gt': total_gt_people,
    #             'detection_rate': total_detections / total_gt_people if total_gt_people > 0 else 0
    #         }
    #     else:
    #         print("\n⚠ No detections made!")
    #         metrics = {
    #             'model': model_name,
    #             'AP': 0,
    #             'AP_50': 0,
    #             'AP_75': 0,
    #             'AP_medium': 0,
    #             'AP_large': 0,
    #             'AR': 0,
    #             'avg_time_ms': avg_time,
    #             'fps': fps,
    #             'total_detections': 0,
    #             'total_gt': total_gt_people,
    #             'detection_rate': 0
    #         }

    #     return metrics

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
            # This matches your original logic and prevents "dropping" detection rate
            valid_anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
            total_gt_people += len(valid_anns)

        return {
            'model': model_name,
            'AP': stats[0],
            'AP_50': stats[1],
            'AP_75': stats[2],
            'AR': stats[5],
            'fps': fps,
            'avg_time_ms': avg_time,
            'total_detections': len(results),
            'total_gt': total_gt_people,
            'detection_rate': len(results) / total_gt_people if total_gt_people > 0 else 0
        }

    def compare_models(self, models_dict, num_images=500):
        """
        Compare multiple models
        """
        all_metrics = []
        
        for model_name, model_config in models_dict.items():
            model, use_gt_boxes = model_config
            # PASS THE ARGUMENT DOWN HERE
            metrics = self.evaluate_model(model, num_images, model_name, use_gt_boxes=use_gt_boxes)
            all_metrics.append(metrics)
        
        # Create comparison visualizations
        self._plot_comparison(all_metrics)
        
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
    
    def _plot_comparison(self, metrics_list):
        """Create comparison plots"""
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
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to 'model_comparison.png'")
        plt.show()

    # # Add this inside class PoseEvaluator in testing.py
    # def visualize_predictions(self, model, num_images=20, output_dir='debug_vis', use_gt_boxes=True):
    #     """
    #     Runs the model and saves images with GT (Green) vs Prediction (Red) overlay.
    #     """
    #     import shutil
        
    #     # 1. Setup Directories
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir)
    #     os.makedirs(output_dir)
        
    #     print(f"(!) Generating {num_images} debug images in '{output_dir}/'...")
        
    #     # COCO Skeleton topology (0-based index)
    #     # Pairs of indices to connect
    #     SKELETON = [
    #         (15,13), (13,11), (16,14), (14,12), # Legs
    #         (11,12), (5,11), (6,12),            # Hips & Torso
    #         (5,6), (5,7), (6,8),                # Shoulders
    #         (7,9), (8,10),                      # Arms
    #         (1,2), (0,1), (0,2), (1,3), (2,4)   # Face
    #     ]
        
    #     person_cat_id = self.coco_gt.getCatIds(catNms=['person'])[0]
    #     subset_ids = self.img_ids[:num_images]

    #     for i, img_id in enumerate(tqdm(subset_ids)):
    #         # Load Image
    #         img_info = self.coco_gt.loadImgs(img_id)[0]
    #         img_path = os.path.join(self.images_path, img_info['file_name'])
    #         original_img = cv2.imread(img_path)
            
    #         if original_img is None: continue
            
    #         vis_img = original_img.copy()
            
    #         # --- 1. DRAW GROUND TRUTH (GREEN) ---
    #         ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False)
    #         anns = self.coco_gt.loadAnns(ann_ids)
            
    #         gt_boxes_list = []
            
    #         for ann in anns:
    #             bbox = ann['bbox'] # x,y,w,h
    #             gt_boxes_list.append(bbox)
                
    #             # Draw GT Skeleton
    #             kp = ann['keypoints'] # [x,y,v, x,y,v...]
                
    #             # Draw Lines
    #             for p1, p2 in SKELETON:
    #                 # Index in list is i*3
    #                 x1, y1, v1 = kp[p1*3], kp[p1*3+1], kp[p1*3+2]
    #                 x2, y2, v2 = kp[p2*3], kp[p2*3+1], kp[p2*3+2]
                    
    #                 # v=0: not labeled, v=1: labeled but invisible, v=2: visible
    #                 if v1 > 0 and v2 > 0:
    #                     pt1 = (int(x1), int(y1))
    #                     pt2 = (int(x2), int(y2))
    #                     cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2) # GREEN for GT
            
    #         # --- 2. DRAW PREDICTIONS (RED) ---
    #         # We pass GT boxes if requested to isolate MP performance
    #         boxes_to_use = gt_boxes_list if use_gt_boxes else None
            
    #         preds = model.predict(original_img, gt_boxes=boxes_to_use)
            
    #         for p in preds:
    #             kps = p['keypoints'] # Numpy [17, 3]
                
    #             for p1, p2 in SKELETON:
    #                 x1, y1, conf1 = kps[p1]
    #                 x2, y2, conf2 = kps[p2]
                    
    #                 # Draw only if MP is somewhat confident (e.g., > 0.1)
    #                 # Note: We draw even low conf to see "garbage" predictions for debugging
    #                 if conf1 > 0.05 and conf2 > 0.05:
    #                     pt1 = (int(x1), int(y1))
    #                     pt2 = (int(x2), int(y2))
    #                     cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2) # RED for Pred
            
    #         # Save
    #         save_path = os.path.join(output_dir, f"debug_{img_id}.jpg")
    #         cv2.imwrite(save_path, vis_img)
            
    #     print(f"Done! Check the '{output_dir}' folder.")

    def visualize_predictions(self, model, num_images=20, output_dir='debug_vis', use_gt_boxes=True):
        """
        Runs the model and saves images with GT (Green) vs Prediction (Red) overlay.
        Compatible with ALL models (MediaPipe, MoveNet, YOLOv8).
        """
        import shutil
        
        # 1. Setup Directories
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
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
    COCO_ANNOTATIONS = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\annotations_trainval2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\val2017\val2017"
    
    # Initialize evaluator
    evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)
    
    # Import models (make sure to use the updated multi-person version!)
    from yolov8_pose import YOLOv8PoseEstimator
    from movenet import MoveNetEstimator
    from mediapipe_pose import MediaPipePoseEstimator
    
    # Initialize models to compare
    # Format: 'Name': (ModelInstance, use_gt_boxes)
    models_to_test = {
        'YOLOv8s-Pose': (YOLOv8PoseEstimator(model_size='s'), False),
        'YOLOv8m-Pose': (YOLOv8PoseEstimator(model_size='m'), False),
        'MoveNet-Lightning (Standard)': (MoveNetEstimator(variant='lightning'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector
        'MoveNet-Thunder (Standard)': (MoveNetEstimator(variant='thunder'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector

        'MoveNet-Lightning (Oracle)': (MoveNetEstimator(variant='lightning'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector
        'MoveNet-Thunder (Oracle)': (MoveNetEstimator(variant='thunder'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector
        'MediaPipe-C2 (Standard)': (MediaPipePoseEstimator(complexity=2), False),
        'MediaPipe-C2 (Oracle)':   (MediaPipePoseEstimator(complexity=2), True),
    }
    
    # Run comparison
    print("\n" + "="*80)
    print("MULTI-PERSON POSE ESTIMATION EVALUATION")
    print("="*80)
    print("This evaluation detects ALL people in each image (proper COCO protocol)")
    print("Expected AP for YOLOv8s-Pose: 0.65-0.70")
    print("Expected AP for YOLOv8m-Pose: 0.70-0.73")
    print("="*80)
    
    # results = evaluator.compare_models(models_to_test, num_images=len(evaluator.img_ids))
    results = evaluator.compare_models(models_to_test, num_images=50)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete! Results saved to 'evaluation_results.json'")
    print("\nYou should now see:")
    print("  - Much higher AP scores (0.65-0.73)")
    print("  - Detection rate around 90-100%")
    print("  - Total detections >> 500 (multiple people per image)")