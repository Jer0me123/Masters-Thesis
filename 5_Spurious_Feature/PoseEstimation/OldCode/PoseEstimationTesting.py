"""
Quick testing script to try all models on a single image
and compare their outputs visually
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# Import all model classes
# Make sure all previous implementation files are saved as:
# yolov8_pose.py, rtmpose.py, movenet.py, mediapipe_pose.py

from yolov8_pose import YOLOv8PoseEstimator
# from rtmpose import RTMPoseEstimator  
from movenet import MoveNetEstimator
from mediapipe_pose import MediaPipePoseEstimator


# COCO keypoint names for visualization
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton connections
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def visualize_pose(image, keypoints, title="Pose", conf_threshold=0.3):
    """Visualize pose estimation on image"""
    vis_img = image.copy()
    h, w = vis_img.shape[:2]
    
    # Draw skeleton
    for connection in COCO_SKELETON:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
        
        if (keypoints[pt1_idx, 2] > conf_threshold and 
            keypoints[pt2_idx, 2] > conf_threshold):
            
            pt1 = (int(keypoints[pt1_idx, 0]), int(keypoints[pt1_idx, 1]))
            pt2 = (int(keypoints[pt2_idx, 0]), int(keypoints[pt2_idx, 1]))
            
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf > conf_threshold:
            cv2.circle(vis_img, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(vis_img, str(i), (int(x)+5, int(y)+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return vis_img


def test_all_models(image_path):
    """Test all models on a single image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize all models
    models = {
        'YOLOv8n-Pose': YOLOv8PoseEstimator(model_size='n'),
        'YOLOv8s-Pose': YOLOv8PoseEstimator(model_size='s'),
        'YOLOv8m-Pose': YOLOv8PoseEstimator(model_size='m'),
        # 'RTMPose-S': RTMPoseEstimator(model_size='s'),
        'MoveNet-Lightning': MoveNetEstimator(variant='lightning'),
        'MoveNet-Thunder': MoveNetEstimator(variant='thunder'),
        'MediaPipe-C0': MediaPipePoseEstimator(complexity=0),
        'MediaPipe-C1': MediaPipePoseEstimator(complexity=1),
    }
    
    results = {}
    visualizations = {}
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE ON ALL MODELS")
    print("="*80)
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        # Warmup
        for _ in range(3):
            _ = model.predict(image)
        
        # Benchmark
        num_runs = 50
        start = time.time()
        for _ in range(num_runs):
            result = model.predict(image)
        end = time.time()
        
        avg_time = (end - start) / num_runs * 1000  # ms
        fps = 1000.0 / avg_time
        
        if result:
            results[model_name] = {
                'prediction': result,
                'avg_time_ms': avg_time,
                'fps': fps,
                'score': result['score']
            }
            
            # Create visualization
            vis_img = visualize_pose(image, result['keypoints'], model_name)
            visualizations[model_name] = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            print(f"  ✓ Detection score: {result['score']:.3f}")
            print(f"  ⚡ Speed: {fps:.1f} FPS ({avg_time:.2f} ms)")
        else:
            print(f"  ✗ No detection")
            results[model_name] = {
                'prediction': None,
                'avg_time_ms': avg_time,
                'fps': fps,
                'score': 0.0
            }
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Detected':>10} {'Score':>10} {'FPS':>10} {'Time (ms)':>12}")
    print("-"*80)
    
    for model_name, res in sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True):
        detected = "✓" if res['prediction'] else "✗"
        score_str = f"{res['score']:.3f}" if res['prediction'] else "N/A"
        print(f"{model_name:<25} {detected:>10} {score_str:>10} {res['fps']:>10.1f} {res['avg_time_ms']:>12.2f}")
    
    # Create comparison grid
    if visualizations:
        n_models = len(visualizations)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, vis_img) in enumerate(visualizations.items()):
            axes[idx].imshow(vis_img)
            axes[idx].set_title(f"{model_name}\n{results[model_name]['fps']:.1f} FPS", 
                               fontsize=10)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(visualizations), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_comparison_visual.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Visual comparison saved to 'model_comparison_visual.png'")
        plt.show()
    
    return results, visualizations


def find_best_model(results):
    """Recommend best model based on speed and accuracy"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Filter out failed detections
    valid_results = {k: v for k, v in results.items() if v['prediction']}
    
    if not valid_results:
        print("❌ No models successfully detected a person")
        return
    
    # Find fastest
    fastest = max(valid_results.items(), key=lambda x: x[1]['fps'])
    print(f"\n🚀 Fastest Model: {fastest[0]}")
    print(f"   Speed: {fastest[1]['fps']:.1f} FPS")
    print(f"   Score: {fastest[1]['score']:.3f}")
    
    # Find highest score
    most_confident = max(valid_results.items(), key=lambda x: x[1]['score'])
    print(f"\n🎯 Most Confident Detection: {most_confident[0]}")
    print(f"   Score: {most_confident[1]['score']:.3f}")
    print(f"   Speed: {most_confident[1]['fps']:.1f} FPS")
    
    # Recommended for RTX 3060 Ti
    print(f"\n💡 Recommended for your RTX 3060 Ti:")
    print(f"   For best speed/accuracy: YOLOv8s-Pose or YOLOv8m-Pose")
    print(f"   For maximum speed: YOLOv8n-Pose or MoveNet-Lightning")
    print(f"   For best accuracy: YOLOv8m-Pose or RTMPose-M")


if __name__ == "__main__":
    # Test on your image
    # Replace with your image path
    image_path = r"G:\Thesis\ImageRetrieval\Professions_125k_Cleaned\Female_Actor\0.208_0000_7336830.jpg"
    
    # You can also download a sample COCO image:
    # wget http://images.cocodataset.org/val2017/000000000139.jpg -O test_image.jpg
    
    results, visualizations = test_all_models(image_path)
    find_best_model(results)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. For full evaluation on COCO, use the evaluation script with:")
    print("   - Download COCO val2017 dataset")
    print("   - Run: python pose_evaluation.py")
    print("\n2. Choose your model and integrate into your pipeline")
    print("\n3. For production, consider:")
    print("   - Batch processing for multiple images")
    print("   - TensorRT optimization for extra speed")
    print("   - Multi-threading for video processing")