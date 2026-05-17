"""
YOLOv8 POSE ESTIMATION HYPERPARAMETER TUNING SUITE
------------------------------------------------------------------------------
Tunes confidence threshold, IOU threshold, and max detections for YOLOv8 models
------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import product
import pandas as pd


class YOLOv8HyperparameterTuner:
    """
    Hyperparameter tuning for YOLOv8 pose estimation models on COCO validation set
    """

    def __init__(self, coco_path, images_path):
        self.coco_path = coco_path
        self.images_path = images_path
        self.coco_gt = COCO(coco_path)

        # All images containing people
        cat_ids = self.coco_gt.getCatIds(catNms=['person'])
        self.img_ids = self.coco_gt.getImgIds(catIds=cat_ids)

    def evaluate_with_params(self, model, conf_thresh, iou_thresh, kp_conf_thresh,
                            num_images=None, model_name='YOLOv8'):
        """
        Evaluate model with specific hyperparameters
        """
        results = []
        person_cat_id = self.coco_gt.getCatIds(catNms=['person'])[0]
        if num_images != None:
            subset_img_ids = self.img_ids[:num_images]
        else:
            subset_img_ids = self.img_ids

        # Set model parameters
        model.model.conf = conf_thresh
        model.model.iou = iou_thresh

        start_time = time.time()
        detections_per_image = {}

        for img_id in tqdm(subset_img_ids, desc=f"conf={conf_thresh:.2f}, iou={iou_thresh:.2f}"):
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_path, img_info['file_name'])
            image = cv2.imread(img_path)

            if image is None:
                continue

            try:
                predictions = model.predict(image)
            except Exception as e:
                print(f"Error predicting on image {img_id}: {e}")
                predictions = []

            if predictions is None:
                predictions = []

            detections_per_image[img_id] = len(predictions)

            for pred in predictions:
                keypoints = pred['keypoints']
                score = pred['score']
                # keypoints_flat = keypoints.flatten().tolist()

                keypoints_xyv = []

                for x, y, c in keypoints:
                    if c >= kp_conf_thresh:
                        keypoints_xyv.extend([float(x), float(y), 2])
                    else:
                        keypoints_xyv.extend([0.0, 0.0, 0])


                bbox = (
                    pred.get('bbox', [0, 0, 0, 0]).tolist()
                    if isinstance(pred.get('bbox'), np.ndarray)
                    else [0, 0, 0, 0]
                )

                results.append({
                    'image_id': img_id,
                    'category_id': person_cat_id,
                    'keypoints': keypoints_xyv,
                    'score': score,
                    'bbox': bbox
                })

        total_time = time.time() - start_time

        if not results:
            return {
                'conf_thresh': conf_thresh,
                'iou_thresh': iou_thresh,
                'AP': 0.0,
                'AP_50': 0.0,
                'AP_75': 0.0,
                'AR': 0.0,
                'fps': 0.0,
                'total_detections': 0,
                'image_detection_rate': 0.0
            }

        # COCO evaluation
        coco_dt = self.coco_gt.loadRes(results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = subset_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats

        # Speed metrics
        total_images = len(subset_img_ids)
        fps = total_images / total_time if total_time > 0 else 0.0

        # Image detection rate
        images_with_gt = 0
        images_with_detection = 0

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
            if detections_per_image.get(img_id, 0) > 0:
                images_with_detection += 1

        image_detection_rate = (
            images_with_detection / images_with_gt
            if images_with_gt > 0 else 0.0
        )

        return {
            'conf_thresh': conf_thresh,
            'iou_thresh': iou_thresh,
            'kp_conf_thresh': kp_conf_thresh,
            'AP': stats[0],
            'AP_50': stats[1],
            'AP_75': stats[2],
            'AR': stats[5],
            'fps': fps,
            'total_detections': len(results),
            'image_detection_rate': image_detection_rate
        }

    def grid_search(self, model, model_name='YOLOv8', num_images=None,
                   conf_thresholds=[0.1, 0.25, 0.4, 0.5, 0.6],
                   iou_thresholds=[0.3, 0.45, 0.6, 0.7],
                   kp_conf_thresholds=[0.3, 0.5, 0.7],
                   output_dir="tuning_results"):
        """
        Perform grid search over hyperparameters
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING FOR {model_name}")
        print(f"{'='*80}")
        print(f"Images: {num_images}") if num_images != None else print(f"Evaluating on ALl Images")
        print(f"Confidence thresholds: {conf_thresholds}")
        print(f"IOU thresholds: {iou_thresholds}")
        print(f"Keypoint Confidence thresholds: {kp_conf_thresholds}")
        print(f"Total combinations: {len(conf_thresholds) * len(iou_thresholds) * len(kp_conf_thresholds)}")
        print(f"{'='*80}\n")

        all_results = []
        
        # Grid search
        for conf, iou, kp_conf in product(conf_thresholds, iou_thresholds, kp_conf_thresholds):
            print(f"\nTesting: conf={conf:.2f}, iou={iou:.2f}")
            
            result = self.evaluate_with_params(
                model, conf, iou, kp_conf,
                num_images=num_images, 
                model_name=model_name
            )
            all_results.append(result)
            
            print(f"  → AP: {result['AP']:.4f}, AP@50: {result['AP_50']:.4f}, "
                  f"AR: {result['AR']:.4f}, FPS: {result['fps']:.2f}")

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_results)
        
        # Save results
        csv_path = os.path.join(output_dir, f"{model_name}_tuning_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Find best configurations
        self._print_best_configs(df, model_name)
        
        # Create visualizations
        self._create_visualizations(df, model_name, output_dir)
        
        # Save detailed JSON
        json_path = os.path.join(output_dir, f"{model_name}_tuning_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        return df

    def _print_best_configs(self, df, model_name):
        """
        Print best configurations for different metrics
        """
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATIONS FOR {model_name}")
        print(f"{'='*80}\n")

        metrics = ['AP', 'AP_50', 'AP_75', 'AR', 'fps', 'image_detection_rate']
        
        for metric in metrics:
            best_row = df.loc[df[metric].idxmax()]
            print(f"Best {metric.upper()}:")
            print(f"  Value: {best_row[metric]:.4f}")
            print(f"  conf_thresh: {best_row['conf_thresh']:.2f}")
            print(f"  iou_thresh: {best_row['iou_thresh']:.2f}")
            print()

        # Best balanced configuration (AP + AR)
        df['balanced_score'] = df['AP'] + df['AR']
        best_balanced = df.loc[df['balanced_score'].idxmax()]
        print(f"Best BALANCED (AP + AR):")
        print(f"  AP: {best_balanced['AP']:.4f}, AR: {best_balanced['AR']:.4f}")
        print(f"  conf_thresh: {best_balanced['conf_thresh']:.2f}")
        print(f"  iou_thresh: {best_balanced['iou_thresh']:.2f}")

    def _create_visualizations(self, df, model_name, output_dir):
        """
        Create comprehensive visualization plots
        """
        # 1. Heatmap for each max_det value
        for max_det in df['max_det'].unique():
            df_subset = df[df['max_det'] == max_det]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{model_name} - Hyperparameter Impact (max_det={int(max_det)})', 
                        fontsize=16, fontweight='bold')

            metrics = ['AP', 'AP_50', 'AR', 'image_detection_rate']
            titles = ['Average Precision (AP)', 'AP @ IoU=0.50', 
                     'Average Recall (AR)', 'Image Detection Rate']

            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx // 2, idx % 2]
                pivot = df_subset.pivot(index='conf_thresh', 
                                       columns='iou_thresh', 
                                       values=metric)
                
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': metric})
                ax.set_title(title)
                ax.set_xlabel('IOU Threshold')
                ax.set_ylabel('Confidence Threshold')

            plt.tight_layout()
            output_path = os.path.join(output_dir, 
                                      f'{model_name}_heatmap_maxdet_{int(max_det)}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Heatmap saved: {output_path}")

        # 2. Line plots showing metric trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} - Metric Trends by Confidence Threshold', 
                    fontsize=16, fontweight='bold')

        metrics = ['AP', 'AR', 'fps', 'total_detections']
        titles = ['Average Precision', 'Average Recall', 
                 'Speed (FPS)', 'Total Detections']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for iou in sorted(df['iou_thresh'].unique()):
                df_iou = df[df['iou_thresh'] == iou]
                grouped = df_iou.groupby('conf_thresh')[metric].mean()
                ax.plot(grouped.index, grouped.values, 
                       marker='o', label=f'IOU={iou:.2f}')
            
            ax.set_xlabel('Confidence Threshold')
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_name}_trends.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Trends plot saved: {output_path}")

        # 3. Pareto front (AP vs FPS)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['fps'], df['AP'], 
                           c=df['conf_thresh'], 
                           s=100, cmap='viridis', 
                           alpha=0.6, edgecolors='black')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence Threshold', rotation=270, labelpad=20)
        
        ax.set_xlabel('Speed (FPS)')
        ax.set_ylabel('Average Precision (AP)')
        ax.set_title(f'{model_name} - Accuracy vs Speed Tradeoff')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_name}_pareto.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Pareto plot saved: {output_path}")


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    COCO_ANNOTATIONS = r"EvaluationDatasets\Coco2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"EvaluationDatasets\Coco2017\val2017"

    parser = argparse.ArgumentParser(description='YOLOv8 Pose Hyperparameter Tuning')
    parser.add_argument("--model_size", type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument("--num_images", type=int, default=None,
                       help='Number of images to evaluate')
    parser.add_argument("--output_dir", type=str, default="tuning_results",
                       help='Output directory for results')
    parser.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.25, 0.4, 0.6], 
                        help="List of confidence thresholds")
    parser.add_argument("--iou", type=float, nargs="+", default=[0.3, 0.45, 0.6, 0.7],
                        help="List of IOU thresholds")
    parser.add_argument("--kp_conf", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Keypoint confidence thresholds")

    args = parser.parse_args()

    print(f"\nInitializing tuner...")
    tuner = YOLOv8HyperparameterTuner(COCO_ANNOTATIONS, COCO_IMAGES)

    print(f"Loading YOLOv8{args.model_size}-Pose model...")
    from yolov8_pose import YOLOv8PoseEstimator
    model = YOLOv8PoseEstimator(model_size=args.model_size)

    # Run grid search
    results_df = tuner.grid_search(
        model=model,
        model_name=f'YOLOv8{args.model_size}-Pose',
        num_images=args.num_images,
        conf_thresholds=args.conf,
        iou_thresholds=args.iou,
        kp_conf_thresholds=args.kp_conf,
        output_dir=args.output_dir
    )

    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")