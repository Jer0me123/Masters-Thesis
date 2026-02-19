import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLOv8PoseEstimator:
    def __init__(self, model_size='m'):
        """
        model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
        Recommended: 's' or 'm' for speed/accuracy balance
        """
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        self.model_name = f'YOLOv8{model_size.upper()}-Pose'
        
    def predict(self, image, conf_threshold=0.25):
        """
        Returns COCO format keypoints for ALL detected persons
        This is the correct way for COCO evaluation!
        
        image: BGR image (cv2 format) or path to image
        Returns: list of dicts, each with 'keypoints' (17x3 array), 'score', and 'bbox'
        """
        results = self.model(image, verbose=False, conf=conf_threshold)
        
        if len(results[0].keypoints) == 0:
            return []
            
        boxes = results[0].boxes
        if len(boxes) == 0:
            return []
        
        detections = []
        for i in range(len(boxes)):
            # Keypoints shape: [17, 3] where 3 = (x, y, confidence)
            keypoints = results[0].keypoints.data[i].cpu().numpy()
            person_conf = boxes.conf[i].cpu().numpy()
            bbox = boxes.xyxy[i].cpu().numpy()
            
            detections.append({
                'keypoints': keypoints,
                'score': float(person_conf),
                'bbox': bbox
            })
        
        return detections
    
    def predict_single(self, image):
        """
        Returns COCO format keypoints for single person (highest confidence)
        Use this for single-person applications
        """
        all_detections = self.predict(image)
        if len(all_detections) == 0:
            return None
        
        # Return highest confidence detection
        return max(all_detections, key=lambda x: x['score'])
    
    def predict_batch(self, images, conf_threshold=0.25):
        """Process multiple images efficiently"""
        results = self.model(images, verbose=False, conf=conf_threshold)
        
        batch_predictions = []
        for result in results:
            if len(result.keypoints) == 0 or len(result.boxes) == 0:
                batch_predictions.append([])
                continue
            
            detections = []
            for i in range(len(result.boxes)):
                keypoints = result.keypoints.data[i].cpu().numpy()
                person_conf = result.boxes.conf[i].cpu().numpy()
                bbox = result.boxes.xyxy[i].cpu().numpy()
                
                detections.append({
                    'keypoints': keypoints,
                    'score': float(person_conf),
                    'bbox': bbox
                })
            
            batch_predictions.append(detections)
        
        return batch_predictions
    
    def benchmark(self, image, num_runs=100):
        """Measure inference speed"""
        # Warmup
        for _ in range(10):
            self.predict(image)
        
        start = time.time()
        for _ in range(num_runs):
            self.predict(image)
        end = time.time()
        
        avg_time = (end - start) / num_runs
        fps = 1.0 / avg_time
        
        return {
            'model': self.model_name,
            'avg_time_ms': avg_time * 1000,
            'fps': fps
        }


# Example usage
if __name__ == "__main__":
    # Initialize model
    estimator = YOLOv8PoseEstimator(model_size='s')
    
    # Load test image
    image = cv2.imread('test_image.jpg')
    
    # Multi-person prediction (for COCO evaluation)
    all_people = estimator.predict(image)
    print(f"Detected {len(all_people)} people")
    for i, person in enumerate(all_people):
        print(f"  Person {i+1}: confidence {person['score']:.3f}")
    
    # Single person prediction (for single-person applications)
    best_person = estimator.predict_single(image)
    if best_person:
        print(f"\nBest person: confidence {best_person['score']:.3f}")
    
    # Visualize all
    annotated = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for idx, person in enumerate(all_people):
        color = colors[idx % len(colors)]
        
        # Draw keypoints
        for kp in person['keypoints']:
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
        
        # Draw bbox
        x1, y1, x2, y2 = person['bbox']
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(annotated, f"Person {idx+1}", (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('YOLOv8 All People', annotated)
    cv2.waitKey(0)
    
    # Benchmark
    benchmark_results = estimator.benchmark(image)
    print(f"\nBenchmark Results:")
    print(f"Model: {benchmark_results['model']}")
    print(f"Average Time: {benchmark_results['avg_time_ms']:.2f} ms")
    print(f"FPS: {benchmark_results['fps']:.2f}")