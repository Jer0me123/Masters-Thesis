# import cv2
# import numpy as np
# import json
# import os
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import time
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# class PoseEvaluator:
#     """
#     Evaluate pose estimation models on COCO validation set
#     """
#     def __init__(self, coco_path, images_path):
#         """
#         coco_path: path to COCO annotations JSON (person_keypoints_val2017.json)
#         images_path: path to COCO val2017 images folder
#         """
#         self.coco_path = coco_path
#         self.images_path = images_path
#         self.coco_gt = COCO(coco_path)
        
#         # Get all images with person annotations
#         cat_ids = self.coco_gt.getCatIds(catNms=['person'])
#         self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)
        
#     # def evaluate_model(self, model, num_images=500, model_name='Model'):
#     #     """
#     #     Evaluate a model on COCO dataset
        
#     #     model: should have a predict(image) method that returns:
#     #            {'keypoints': np.array([17, 3]), 'score': float, 'bbox': np.array([4])}
#     #     num_images: number of images to evaluate on (max)
#     #     """
#     #     print(f"\n{'='*60}")
#     #     print(f"Evaluating {model_name}")
#     #     print(f"{'='*60}")
        
#     #     # Limit number of images
#     #     test_img_ids = self.img_ids[:num_images]
        
#     #     results = []
#     #     inference_times = []
#     #     failed_count = 0
        
#     #     for img_id in tqdm(test_img_ids, desc=f"Processing {model_name}"):
#     #         # Load image
#     #         img_info = self.coco_gt.loadImgs(img_id)[0]
#     #         img_path = os.path.join(self.images_path, img_info['file_name'])
            
#     #         if not os.path.exists(img_path):
#     #             continue
                
#     #         image = cv2.imread(img_path)
#     #         if image is None:
#     #             continue
            
#     #         # Get ground truth annotations
#     #         ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
#     #         anns = self.coco_gt.loadAnns(ann_ids)
            
#     #         if len(anns) == 0:
#     #             continue
            
#     #         # For single-person: find annotation with largest area
#     #         if len(anns) > 1:
#     #             areas = [ann['area'] for ann in anns]
#     #             main_ann_idx = np.argmax(areas)
#     #         else:
#     #             main_ann_idx = 0
            
#     #         # Run inference
#     #         start_time = time.time()
#     #         pred = model.predict(image)
#     #         inference_time = time.time() - start_time
#     #         inference_times.append(inference_time)
            
#     #         if pred is None:
#     #             failed_count += 1
#     #             continue
            
#     #         # Format prediction for COCO evaluation
#     #         keypoints_flat = pred['keypoints'].flatten().tolist()
            
#     #         result = {
#     #             'image_id': img_id,
#     #             'category_id': 1,
#     #             'keypoints': keypoints_flat,
#     #             'score': pred['score']
#     #         }
#     #         results.append(result)
        
#     #     # Calculate speed metrics
#     #     avg_time = np.mean(inference_times) * 1000  # ms
#     #     fps = 1000.0 / avg_time
        
#     #     print(f"\nSpeed Metrics:")
#     #     print(f"  Average inference time: {avg_time:.2f} ms")
#     #     print(f"  FPS: {fps:.2f}")
#     #     print(f"  Failed detections: {failed_count}/{len(test_img_ids)}")
        
#     #     # Save results for COCO evaluation
#     #     results_file = f'{model_name.replace(" ", "_")}_results.json'
#     #     with open(results_file, 'w') as f:
#     #         json.dump(results, f)
        
#     #     # Run COCO evaluation
#     #     if len(results) > 0:
#     #         print(f"\nAccuracy Metrics (COCO AP):")
#     #         coco_dt = self.coco_gt.loadRes(results_file)
#     #         coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
#     #         coco_eval.params.imgIds = test_img_ids
#     #         coco_eval.evaluate()
#     #         coco_eval.accumulate()
#     #         coco_eval.summarize()
            
#     #         # Extract key metrics
#     #         metrics = {
#     #             'model': model_name,
#     #             'AP': coco_eval.stats[0],  # Average Precision (AP) @[IoU=0.50:0.95]
#     #             'AP_50': coco_eval.stats[1],  # AP @[IoU=0.50]
#     #             'AP_75': coco_eval.stats[2],  # AP @[IoU=0.75]
#     #             'AR': coco_eval.stats[5],  # Average Recall (AR)
#     #             'avg_time_ms': avg_time,
#     #             'fps': fps,
#     #             'failed_rate': failed_count / len(test_img_ids)
#     #         }
#     #     else:
#     #         metrics = {
#     #             'model': model_name,
#     #             'AP': 0,
#     #             'AP_50': 0,
#     #             'AP_75': 0,
#     #             'AR': 0,
#     #             'avg_time_ms': avg_time,
#     #             'fps': fps,
#     #             'failed_rate': 1.0
#     #         }
        
#     #     return metrics

#     def evaluate_model(self, model, num_images=500, model_name='Model'):
#         print(f"\n{'='*60}")
#         print(f"Evaluating {model_name}")
#         print(f"{'='*60}")

#         test_img_ids = self.img_ids[:num_images]

#         results = []
#         inference_times = []
#         failed_count = 0

#         for img_id in tqdm(test_img_ids, desc=f"Processing {model_name}"):
#             img_info = self.coco_gt.loadImgs(img_id)[0]
#             img_path = os.path.join(self.images_path, img_info['file_name'])

#             image = cv2.imread(img_path)
#             if image is None:
#                 continue

#             ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
#             anns = self.coco_gt.loadAnns(ann_ids)
#             # if len(anns) == 0:
#             #     continue

#             # COCO keypoint eval requires num_keypoints > 0
#             anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
#             if len(anns) == 0:
#                 continue

#             # single-person protocol: largest visible-keypoint person
#             main_ann = max(anns, key=lambda a: a['area'])

#             # print(main_ann['num_keypoints'])
#             gt_bbox = main_ann['bbox']  # [x,y,w,h]

#             start = time.time()

#             # Top-down models need cropping
#             if getattr(model, "requires_crop", False):
#                 x, y, w, h = map(int, gt_bbox)
#                 crop = image[y:y+h, x:x+w]
#                 pred = model.predict(crop)
#                 if pred is not None:
#                     pred['keypoints'][:, 0] += x
#                     pred['keypoints'][:, 1] += y
#             else:
#                 pred = model.predict(image)

#             inference_times.append(time.time() - start)

#             if pred is None:
#                 failed_count += 1
#                 keypoints = [0, 0, 0] * 17
#                 bbox = [0, 0, 1, 1]
#                 score = 0.001
#             else:
#                 # visibility conversion
#                 kp = pred['keypoints'].copy()
#                 conf = kp[:, 2]
#                 kp[:, 2] = np.where(conf > 0.5, 2,
#                                     np.where(conf > 0.1, 1, 0))
#                 keypoints = kp.flatten().tolist()

#                 x1, y1, x2, y2 = pred['bbox']
#                 bbox = [x1, y1, x2 - x1, y2 - y1]
#                 score = float(pred['score'])

#             results.append({
#                 'image_id': int(img_id),
#                 'category_id': 1,
#                 'keypoints': [float(x) for x in keypoints],
#                 'bbox': [float(x) for x in bbox],
#                 'score': float(score)
#             })


#         avg_time = np.mean(inference_times) * 1000
#         fps = 1000.0 / avg_time

#         print(f"\nSpeed Metrics:")
#         print(f"  Average inference time: {avg_time:.2f} ms")
#         print(f"  FPS: {fps:.2f}")
#         print(f"  Failed detections: {failed_count}/{len(test_img_ids)}")

#         results_file = f'{model_name.replace(" ", "_")}_results.json'
#         with open(results_file, 'w') as f:
#             json.dump(results, f)

#         coco_dt = self.coco_gt.loadRes(results_file)
#         coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
#         coco_eval.params.imgIds = test_img_ids
#         coco_eval.params.useCats = 1
#         coco_eval.params.catIds = [1]
#         # coco_eval.params.maxDets = [1]

#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()

#         return {
#             'model': model_name,
#             'AP': coco_eval.stats[0],
#             'AP_50': coco_eval.stats[1],
#             'AP_75': coco_eval.stats[2],
#             'AR': coco_eval.stats[5],
#             'avg_time_ms': avg_time,
#             'fps': fps,
#             'failed_rate': failed_count / len(test_img_ids)
#         }

    
#     def compare_models(self, models_dict, num_images=500):
#         """
#         Compare multiple models
        
#         models_dict: {'Model Name': model_instance, ...}
#         """
#         all_metrics = []
        
#         for model_name, model in models_dict.items():
#             metrics = self.evaluate_model(model, num_images, model_name)
#             all_metrics.append(metrics)
        
#         # Create comparison visualizations
#         self._plot_comparison(all_metrics)
        
#         # Print summary table
#         self._print_summary_table(all_metrics)
        
#         return all_metrics
    
#     def _print_summary_table(self, metrics_list):
#         """Print comparison table"""
#         print(f"\n{'='*100}")
#         print(f"COMPARISON SUMMARY")
#         print(f"{'='*100}")
#         print(f"{'Model':<25} {'AP':>8} {'AP@50':>8} {'AP@75':>8} {'AR':>8} {'FPS':>8} {'Time(ms)':>10}")
#         print(f"{'-'*100}")
        
#         for m in sorted(metrics_list, key=lambda x: x['AP'], reverse=True):
#             print(f"{m['model']:<25} {m['AP']:>8.3f} {m['AP_50']:>8.3f} {m['AP_75']:>8.3f} "
#                   f"{m['AR']:>8.3f} {m['fps']:>8.1f} {m['avg_time_ms']:>10.2f}")
    
#     def _plot_comparison(self, metrics_list):
#         """Create comparison plots"""
#         fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
#         models = [m['model'] for m in metrics_list]
#         aps = [m['AP'] for m in metrics_list]
#         fps_values = [m['fps'] for m in metrics_list]
        
#         # AP comparison
#         axes[0].barh(models, aps, color='skyblue')
#         axes[0].set_xlabel('Average Precision (AP)')
#         axes[0].set_title('Accuracy Comparison')
#         axes[0].set_xlim(0, 1)
        
#         # FPS comparison
#         axes[1].barh(models, fps_values, color='lightcoral')
#         axes[1].set_xlabel('Frames Per Second (FPS)')
#         axes[1].set_title('Speed Comparison')
        
#         plt.tight_layout()
#         plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
#         print(f"\nComparison plot saved to 'model_comparison.png'")
#         plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Setup paths (adjust these to your dataset location)
#     COCO_ANNOTATIONS = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\annotations_trainval2017\annotations\person_keypoints_val2017.json"
#     COCO_IMAGES = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\val2017\val2017"
    
#     # Initialize evaluator
#     evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)
    
#     # Import all model classes (assuming they're in separate files)
#     from yolov8_pose import YOLOv8PoseEstimator
#     # from rtmpose import RTMPoseEstimator
#     from movenet import MoveNetEstimator
#     from mediapipe_pose import MediaPipePoseEstimator
    
#     # Initialize models to compare
#     models_to_test = {
#         'YOLOv8s-Pose': YOLOv8PoseEstimator(model_size='s'),
#         # 'YOLOv8m-Pose': YOLOv8PoseEstimator(model_size='m'),
#         # 'RTMPose-S': RTMPoseEstimator(model_size='s'),
#         # 'RTMPose-M': RTMPoseEstimator(model_size='m'),
#         # 'MoveNet-Lightning': MoveNetEstimator(variant='lightning'),
#         # 'MoveNet-Thunder': MoveNetEstimator(variant='thunder'),
#         # 'MediaPipe-C1': MediaPipePoseEstimator(complexity=1),
#     }
    
#     # Run comparison (test on 500 images - adjust as needed)
#     # For full evaluation, use num_images=len(evaluator.img_ids)
#     results = evaluator.compare_models(models_to_test, num_images=500)
    
#     # Save results
#     with open('evaluation_results.json', 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print("\n✅ Evaluation complete! Results saved to 'evaluation_results.json'")


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
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        box1, box2: [x1, y1, x2, y2] format
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def xywh_to_xyxy(self, bbox):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    
    def evaluate_model(self, model, num_images=500, model_name='Model'):
        """
        Evaluate model using proper detection matching
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")

        test_img_ids = self.img_ids[:num_images]

        results = []
        inference_times = []
        failed_count = 0
        
        # For debugging
        match_stats = {'exact_match': 0, 'iou_match': 0, 'no_match': 0}

        for img_id in tqdm(test_img_ids, desc=f"Processing {model_name}"):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_path, img_info['file_name'])

            image = cv2.imread(img_path)
            if image is None:
                continue

            # Get all person annotations with keypoints
            ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
            anns = self.coco_gt.loadAnns(ann_ids)
            anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
            
            if len(anns) == 0:
                continue

            # Run inference
            start = time.time()
            pred = model.predict(image)
            inference_times.append(time.time() - start)

            if pred is None:
                failed_count += 1
                # Still need to submit result for proper evaluation
                # Use the largest annotation as target
                main_ann = max(anns, key=lambda a: a['area'])
                results.append({
                    'image_id': int(img_id),
                    'category_id': 1,
                    'keypoints': [0.0] * 51,  # 17 keypoints * 3
                    'bbox': main_ann['bbox'],
                    'score': 0.001
                })
                match_stats['no_match'] += 1
                continue

            # Convert prediction bbox to xyxy format
            pred_bbox_xyxy = pred['bbox']  # Already in xyxy format from YOLO
            
            # Find best matching ground truth annotation
            best_iou = 0
            best_ann = None
            
            for ann in anns:
                gt_bbox_xyxy = self.xywh_to_xyxy(ann['bbox'])
                iou = self.compute_iou(pred_bbox_xyxy, gt_bbox_xyxy)
                
                if iou > best_iou:
                    best_iou = iou
                    best_ann = ann
            
            # Use the matched annotation (or largest if no good match)
            if best_ann is None or best_iou < 0.3:
                best_ann = max(anns, key=lambda a: a['area'])
                match_stats['no_match'] += 1
            elif best_iou > 0.7:
                match_stats['exact_match'] += 1
            else:
                match_stats['iou_match'] += 1

            # Convert keypoint confidences to COCO visibility format
            kp = pred['keypoints'].copy()
            conf = kp[:, 2]
            # 0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible
            kp[:, 2] = np.where(conf > 0.5, 2, np.where(conf > 0.1, 1, 0))
            keypoints = kp.flatten().tolist()

            # Convert bbox to xywh format for COCO
            x1, y1, x2, y2 = pred['bbox']
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            results.append({
                'image_id': int(img_id),
                'category_id': 1,
                'keypoints': [float(x) for x in keypoints],
                'bbox': bbox,
                'score': float(pred['score'])
            })

        # Print matching statistics
        total = sum(match_stats.values())
        print(f"\nMatching Statistics:")
        print(f"  Exact matches (IoU > 0.7): {match_stats['exact_match']}/{total} ({100*match_stats['exact_match']/total:.1f}%)")
        print(f"  Partial matches (IoU 0.3-0.7): {match_stats['iou_match']}/{total} ({100*match_stats['iou_match']/total:.1f}%)")
        print(f"  No match/Failed: {match_stats['no_match']}/{total} ({100*match_stats['no_match']/total:.1f}%)")

        avg_time = np.mean(inference_times) * 1000
        fps = 1000.0 / avg_time

        print(f"\nSpeed Metrics:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Failed detections: {failed_count}/{len(test_img_ids)}")

        # Save results
        results_file = f'{model_name.replace(" ", "_")}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)

        # Run COCO evaluation
        print(f"\nAccuracy Metrics (COCO AP):")
        coco_dt = self.coco_gt.loadRes(results_file)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = test_img_ids
        coco_eval.params.useCats = 1
        coco_eval.params.catIds = [1]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            'model': model_name,
            'AP': coco_eval.stats[0],
            'AP_50': coco_eval.stats[1],
            'AP_75': coco_eval.stats[2],
            'AR': coco_eval.stats[5],
            'avg_time_ms': avg_time,
            'fps': fps,
            'failed_rate': failed_count / len(test_img_ids),
            'match_rate': (match_stats['exact_match'] + match_stats['iou_match']) / total
        }
    
    def compare_models(self, models_dict, num_images=500):
        """
        Compare multiple models
        
        models_dict: {'Model Name': model_instance, ...}
        """
        all_metrics = []
        
        for model_name, model in models_dict.items():
            metrics = self.evaluate_model(model, num_images, model_name)
            all_metrics.append(metrics)
        
        # Create comparison visualizations
        self._plot_comparison(all_metrics)
        
        # Print summary table
        self._print_summary_table(all_metrics)
        
        return all_metrics
    
    def _print_summary_table(self, metrics_list):
        """Print comparison table"""
        print(f"\n{'='*110}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*110}")
        print(f"{'Model':<25} {'AP':>8} {'AP@50':>8} {'AP@75':>8} {'AR':>8} {'FPS':>8} {'Time(ms)':>10} {'Match%':>8}")
        print(f"{'-'*110}")
        
        for m in sorted(metrics_list, key=lambda x: x['AP'], reverse=True):
            print(f"{m['model']:<25} {m['AP']:>8.3f} {m['AP_50']:>8.3f} {m['AP_75']:>8.3f} "
                  f"{m['AR']:>8.3f} {m['fps']:>8.1f} {m['avg_time_ms']:>10.2f} {m['match_rate']*100:>7.1f}%")
    
    def _plot_comparison(self, metrics_list):
        """Create comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        models = [m['model'] for m in metrics_list]
        aps = [m['AP'] for m in metrics_list]
        fps_values = [m['fps'] for m in metrics_list]
        
        # AP comparison
        axes[0].barh(models, aps, color='skyblue')
        axes[0].set_xlabel('Average Precision (AP)')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xlim(0, 1)
        
        # FPS comparison
        axes[1].barh(models, fps_values, color='lightcoral')
        axes[1].set_xlabel('Frames Per Second (FPS)')
        axes[1].set_title('Speed Comparison')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to 'model_comparison.png'")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Setup paths
    COCO_ANNOTATIONS = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\annotations_trainval2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\val2017\val2017"
    
    # Initialize evaluator
    evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)
    
    # Import models
    from yolov8_pose import YOLOv8PoseEstimator
    from movenet import MoveNetEstimator
    from mediapipe_pose import MediaPipePoseEstimator
    
    # Initialize models to compare
    models_to_test = {
        'YOLOv8s-Pose': YOLOv8PoseEstimator(model_size='s'),
        'YOLOv8m-Pose': YOLOv8PoseEstimator(model_size='m'),
        # 'MoveNet-Lightning': MoveNetEstimator(variant='lightning'),
        # 'MediaPipe-C1': MediaPipePoseEstimator(complexity=1),
    }
    
    # Run comparison
    results = evaluator.compare_models(models_to_test, num_images=500)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete! Results saved to 'evaluation_results.json'")