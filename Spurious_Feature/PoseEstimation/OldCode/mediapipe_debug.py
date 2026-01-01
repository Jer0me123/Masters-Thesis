"""
Debug script to visualize MediaPipe keypoints and diagnose coordinate issues
"""
import cv2
import numpy as np
from mediapipe_pose import MediaPipePoseEstimator
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os

def visualize_predictions(image, predictions, gt_annotations=None, title="MediaPipe Predictions"):
    """
    Visualize predictions vs ground truth
    """
    fig, axes = plt.subplots(1, 2 if gt_annotations else 1, figsize=(12 if gt_annotations else 6, 6))
    if not gt_annotations:
        axes = [axes]
    
    # Show predictions
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Predictions ({len(predictions)} people)')
    
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    
    for idx, pred in enumerate(predictions):
        color = colors[idx % len(colors)]
        
        print(f"\n  Person {idx+1}:")
        print(f"    Score: {pred['score']:.3f}")
        print(f"    Bbox: {pred['bbox']}")
        
        # Draw bbox
        x1, y1, x2, y2 = pred['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor=color, linewidth=2)
        axes[0].add_patch(rect)
        
        # Draw keypoints
        keypoints = pred['keypoints']
        valid_count = 0
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                axes[0].plot(x, y, 'o', color=color, markersize=6)
                axes[0].text(x+3, y-3, str(i), color='white', fontsize=7,
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
                valid_count += 1
                
                # Print sample keypoints
                if i in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
                    print(f"      KP {i}: x={x:.1f}, y={y:.1f}, conf={conf:.3f}")
        
        print(f"    Valid keypoints: {valid_count}/17")
    
    axes[0].axis('off')
    
    # Show ground truth if provided
    if gt_annotations:
        axes[1].imshow(img_rgb)
        axes[1].set_title(f'Ground Truth ({len(gt_annotations)} people)')
        
        for idx, ann in enumerate(gt_annotations):
            color = colors[idx % len(colors)]
            
            # Draw bbox
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, 
                                 fill=False, edgecolor=color, linewidth=2)
            axes[1].add_patch(rect)
            
            # Draw keypoints
            kps = np.array(ann['keypoints']).reshape(17, 3)
            for i, (x_gt, y_gt, v) in enumerate(kps):
                if v > 0:
                    axes[1].plot(x_gt, y_gt, 'o', color=color, markersize=6)
                    axes[1].text(x_gt+3, y_gt-3, str(i), color='white', fontsize=7,
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def debug_mediapipe(coco_annotations_path, coco_images_path, num_samples=5):
    """
    Debug MediaPipe on sample images
    """
    print("="*80)
    print("MEDIAPIPE DEBUG - COORDINATE VERIFICATION")
    print("="*80)
    
    # Load COCO
    coco = COCO(coco_annotations_path)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    # Initialize MediaPipe
    estimator = MediaPipePoseEstimator(complexity=1)
    
    # Test on sample images
    sample_ids = img_ids[:num_samples]
    
    for img_id in sample_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_images_path, img_info['file_name'])
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        print(f"\n{'='*80}")
        print(f"Image: {img_info['file_name']}")
        print(f"Size: {w}x{h}")
        
        # Get ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        anns = [a for a in anns if a.get('num_keypoints', 0) > 0]
        
        print(f"Ground truth people: {len(anns)}")
        
        # Run prediction
        predictions = estimator.predict(image)
        print(f"Predicted people: {len(predictions)}")
        
        if len(predictions) == 0:
            print("  WARNING: No detections!")
            continue
        
        # Check coordinate validity
        all_valid = True
        for idx, pred in enumerate(predictions):
            keypoints = pred['keypoints']
            
            # Check if keypoints are within image bounds
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            
            out_of_bounds = np.sum((x_coords < 0) | (x_coords > w) | 
                                   (y_coords < 0) | (y_coords > h))
            
            if out_of_bounds > 0:
                print(f"  Person {idx+1}: {out_of_bounds}/17 keypoints OUT OF BOUNDS!")
                print(f"    X range: [{x_coords.min():.1f}, {x_coords.max():.1f}] (image width: {w})")
                print(f"    Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}] (image height: {h})")
                all_valid = False
            
            # Check if keypoints are all zeros (missing)
            zero_count = np.sum(np.all(keypoints == 0, axis=1))
            if zero_count > 10:
                print(f"  Person {idx+1}: {zero_count}/17 keypoints are ZERO!")
                all_valid = False
        
        if all_valid:
            print("  ✓ All keypoints within image bounds")
        
        # Visualize
        fig = visualize_predictions(image, predictions, anns, 
                                    title=f"Image {img_id}")
        plt.savefig(f'mediapipe_debug_{img_id}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: mediapipe_debug_{img_id}.png")
        plt.close()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\nCheck the saved images to verify:")
    print("  1. Are keypoints on the correct body parts?")
    print("  2. Are keypoints inside the person's bounding box?")
    print("  3. Do predictions align with ground truth?")


if __name__ == "__main__":
    COCO_ANNOTATIONS = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\annotations_trainval2017\annotations\person_keypoints_val2017.json"
    COCO_IMAGES = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\val2017\val2017"
    
    debug_mediapipe(COCO_ANNOTATIONS, COCO_IMAGES, num_samples=5)