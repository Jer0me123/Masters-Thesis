"""
MULTI-PERSON POSE ESTIMATION EVALUATION SUITE
------------------------------------------------------------------------------
UPDATED:
- Detection rate is now IMAGE-LEVEL (Image Detection Rate)
- Guaranteed to be <= 100%
- Pose-count-based detection removed
------------------------------------------------------------------------------
"""

import sys
import os

# Add 'pose_models' to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pose_models'))

import cv2
import numpy as np
import json
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class PoseEvaluator:
    """
    Evaluate pose estimation models on COCO validation set
    """

    def __init__(self, coco_path, images_path):
        self.coco_path = coco_path
        self.images_path = images_path
        self.coco_gt = COCO(coco_path)

        # All images containing people
        cat_ids = self.coco_gt.getCatIds(catNms=['person'])
        self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)

    def evaluate_model(self, model, num_images=500, model_name='Model', use_gt_boxes=False):
        results = []
        person_cat_id = self.coco_gt.getCatIds(catNms=['person'])[0]

        subset_img_ids = self.img_ids[:num_images]

        print(f"Evaluating {model_name} on {len(subset_img_ids)} images...")
        if use_gt_boxes:
            print("(!) ORACLE MODE: Using Ground Truth boxes")

        start_time = time.time()

        # ------------------------------------------------------------
        # NEW: Track detections per image (for Image Detection Rate)
        # ------------------------------------------------------------
        detections_per_image = {}

        for img_id in tqdm(subset_img_ids):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_path, img_info['file_name'])
            image = cv2.imread(img_path)

            if image is None:
                continue

            current_gt_boxes = None
            if use_gt_boxes:
                ann_ids = self.coco_gt.getAnnIds(
                    imgIds=img_id,
                    catIds=[person_cat_id],
                    iscrowd=False
                )
                anns = self.coco_gt.loadAnns(ann_ids)
                current_gt_boxes = [a['bbox'] for a in anns]

                if not current_gt_boxes:
                    continue

            try:
                if use_gt_boxes and current_gt_boxes is not None:
                    predictions = model.predict(image, gt_boxes=current_gt_boxes)
                else:
                    predictions = model.predict(image)
            except TypeError:
                predictions = model.predict(image)

            if predictions is None:
                predictions = []

            # --------------------------------------------------------
            # Track detections per image (IMAGE-LEVEL metric)
            # --------------------------------------------------------
            detections_per_image[img_id] = len(predictions)

            for pred in predictions:
                keypoints = pred['keypoints']
                score = pred['score']
                keypoints_flat = keypoints.flatten().tolist()

                bbox = (
                    pred.get('bbox', [0, 0, 0, 0]).tolist()
                    if isinstance(pred.get('bbox'), np.ndarray)
                    else [0, 0, 0, 0]
                )

                results.append({
                    'image_id': img_id,
                    'category_id': person_cat_id,
                    'keypoints': keypoints_flat,
                    'score': score,
                    'bbox': bbox
                })

        total_time = time.time() - start_time

        if not results:
            print(f"Warning: {model_name} made 0 detections.")
            return {
                'model': model_name,
                'AP': 0.0,
                'AP_50': 0.0,
                'AP_75': 0.0,
                'AR': 0.0,
                'fps': 0.0,
                'avg_time_ms': 0.0,
                'total_detections': 0,
                'total_gt': 0,
                'image_detection_rate': 0.0
            }

        # ------------------------------------------------------------
        # COCO EVALUATION
        # ------------------------------------------------------------
        coco_dt = self.coco_gt.loadRes(results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = subset_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats

        # ------------------------------------------------------------
        # Speed metrics
        # ------------------------------------------------------------
        total_images = len(subset_img_ids)
        avg_time_ms = (total_time / total_images) * 1000
        fps = total_images / total_time if total_time > 0 else 0.0

        # ------------------------------------------------------------
        # IMAGE DETECTION RATE (NEW, <= 100%)
        # ------------------------------------------------------------
        images_with_gt = 0
        images_with_detection = 0
        total_gt_people = 0

        for img_id in subset_img_ids:
            ann_ids = self.coco_gt.getAnnIds(
                imgIds=img_id,
                catIds=[person_cat_id],
                iscrowd=False
            )
            anns = self.coco_gt.loadAnns(ann_ids)

            valid_anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
            if not valid_anns:
                continue

            images_with_gt += 1
            total_gt_people += len(valid_anns)

            if detections_per_image.get(img_id, 0) > 0:
                images_with_detection += 1

        image_detection_rate = (
            images_with_detection / images_with_gt
            if images_with_gt > 0 else 0.0
        )

        return {
            'model': model_name,
            'AP': stats[0],
            'AP_50': stats[1],
            'AP_75': stats[2],
            'AR': stats[5],
            'fps': fps,
            'avg_time_ms': avg_time_ms,
            'total_detections': len(results),
            'total_gt': total_gt_people,
            'image_detection_rate': image_detection_rate
        }

    def compare_models(self, models_dict, num_images=500, output_dir="."):
        all_metrics = []

        for model_name, (model, use_gt_boxes) in models_dict.items():
            metrics = self.evaluate_model(
                model,
                num_images=num_images,
                model_name=model_name,
                use_gt_boxes=use_gt_boxes
            )
            all_metrics.append(metrics)

        self._plot_comparison(all_metrics, output_dir)
        self._print_summary_table(all_metrics)

        return all_metrics

    def _print_summary_table(self, metrics_list):
        print(f"\n{'='*120}")
        print("COMPARISON SUMMARY")
        print(f"{'='*120}")
        print(
            f"{'Model':<25} {'AP':>8} {'AP@50':>8} {'AP@75':>8} "
            f"{'AR':>8} {'FPS':>8} {'Time(ms)':>10} "
            f"{'Detections':>12} {'Img Det %':>10}"
        )
        print(f"{'-'*120}")

        for m in sorted(metrics_list, key=lambda x: x['AP'], reverse=True):
            det_str = f"{m['total_detections']}/{m['total_gt']}"
            print(
                f"{m['model']:<25} "
                f"{m['AP']:>8.3f} {m['AP_50']:>8.3f} {m['AP_75']:>8.3f} "
                f"{m['AR']:>8.3f} {m['fps']:>8.1f} {m['avg_time_ms']:>10.2f} "
                f"{det_str:>12} {m['image_detection_rate']*100:>9.1f}%"
            )

    def _plot_comparison(self, metrics_list, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "model_comparison.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        models = [m['model'] for m in metrics_list]
        aps = [m['AP'] for m in metrics_list]
        fps_values = [m['fps'] for m in metrics_list]
        img_det_rates = [m['image_detection_rate'] * 100 for m in metrics_list]

        axes[0].barh(models, aps, color='skyblue')
        axes[0].set_xlabel('Average Precision (AP)')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xlim(0, 1)

        axes[1].barh(models, fps_values, color='lightcoral')
        axes[1].set_xlabel('Frames Per Second (FPS)')
        axes[1].set_title('Speed Comparison')

        axes[2].barh(models, img_det_rates, color='lightgreen')
        axes[2].set_xlabel('Image Detection Rate (%)')
        axes[2].set_title('Image-Level Robustness')
        axes[2].set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to '{out_path}'")

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    COCO_ANNOTATIONS = r"EvaluationDatasets\Coco2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"EvaluationDatasets\Coco2017\val2017"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    evaluator = PoseEvaluator(COCO_ANNOTATIONS, COCO_IMAGES)

    from yolov8_pose import YOLOv8PoseEstimator
    from movenet import MoveNetEstimator
    from mediapipe_pose import MediaPipePoseEstimator
    from rtmpose import MMPoseTopDownEstimator

    models_to_test = {
        'YOLOv8n-Pose': (YOLOv8PoseEstimator(model_size='n'), False),
        'YOLOv8s-Pose': (YOLOv8PoseEstimator(model_size='s'), False),
        'YOLOv8m-Pose': (YOLOv8PoseEstimator(model_size='m'), False),
        'YOLOv8l-Pose': (YOLOv8PoseEstimator(model_size='l'), False),
        'YOLOv8x-Pose': (YOLOv8PoseEstimator(model_size='x'), False),

        'MoveNet-Lightning (Standard)': (MoveNetEstimator(variant='lightning'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector
        'MoveNet-Lightning (Oracle)': (MoveNetEstimator(variant='lightning'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector

        'MoveNet-Thunder (Standard)': (MoveNetEstimator(variant='thunder'), False), # CPU bound due to tf / doesn't support batching / requires an external person detector
        'MoveNet-Thunder (Oracle)': (MoveNetEstimator(variant='thunder'), True), # CPU bound due to tf / doesn't support batching / requires an external person detector
        
        'MediaPipe-C2 (Standard)': (MediaPipePoseEstimator(complexity=2), False), # requires an external person detector
        'MediaPipe-C2 (Oracle)':   (MediaPipePoseEstimator(complexity=2), True), # requires an external person detector
        
        "HRNet-W32 (Standard)": (MMPoseTopDownEstimator("w32_256", use_yolo=True), False), # requires an external person detector
        "HRNet-W32 (Oracle)": (MMPoseTopDownEstimator("w32_256", use_yolo=True), True), # requires an external person detector

        "HRNet-W32-384 (Standard - Large)": (MMPoseTopDownEstimator("w32_384", use_yolo=True), False), # requires an external person detector
        "HRNet-W32-384 (Oracle - Large)": (MMPoseTopDownEstimator("w32_384", use_yolo=True), True), # requires an external person detector

        "HRNet-W48 (Standard)": (MMPoseTopDownEstimator("w48_256", use_yolo=True), False), # requires an external person detector
        "HRNet-W48 (Oracle)": (MMPoseTopDownEstimator("w48_256", use_yolo=False), True), # requires an external person detector
    }

    results = evaluator.compare_models(
        models_to_test,
        num_images=len(evaluator.img_ids),
        output_dir=args.output_dir
    )

    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete.")


# .venv_test\Scripts\python.exe PoseModelEvaluation.py --output_dir "EvaluationDatasets\Coco2017"

# NOTE: Evaluation is carried out on images containing people only