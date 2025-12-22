import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from ultralytics import YOLO

class MoveNetEstimator:
    def __init__(self, variant='lightning', yolo_model='yolov8m.pt'):
        """
        variant: 'lightning' (fastest) or 'thunder' (more accurate)
        Note: MoveNet is single-person only, so we use detection + cropping for multi-person
        """
        if variant == 'lightning':
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            self.input_size = 192
        else:
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            self.input_size = 256
        
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        self.model_name = f'MoveNet-{variant.capitalize()}'
        self.requires_crop = True
        
        # Person detector for multi-person support
        print(f"[Info] Loading {yolo_model} for person detection...")
        self.detector = YOLO(yolo_model)
        
        # OPTIMIZATION: Pre-compile TensorFlow graphs
        print(f"[{self.model_name}] Warming up TensorFlow model...")
        dummy_input = tf.zeros((1, self.input_size, self.input_size, 3), dtype=tf.int32)
        _ = self.movenet(dummy_input)
        print(f"[{self.model_name}] Ready!")
        
    def predict(self, image, gt_boxes=None, conf_threshold=0.25, max_detections=5):
        """
        Returns COCO format keypoints for ALL detected persons
        Supports 'gt_boxes' for Oracle Evaluation.
        """
        candidate_boxes = [] # List of tuples: (x1, y1, x2, y2, score)

        # ---------------------------------------------------------
        # MODE 1: USE GROUND TRUTH (Oracle)
        # ---------------------------------------------------------
        if gt_boxes is not None:
            for box in gt_boxes:
                # COCO GT boxes are [x, y, w, h]
                x, y, w, h = box
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                # Assign 1.0 confidence because these are ground truth
                candidate_boxes.append((x1, y1, x2, y2, 1.0))

        # ---------------------------------------------------------
        # MODE 2: USE YOLO DETECTOR
        # ---------------------------------------------------------
        else:
            # Detect all people (fast - ~10ms)
            det_results = self.detector(image, classes=[0], verbose=False, conf=conf_threshold)
            
            if len(det_results[0].boxes) > 0:
                # Limit detections to avoid extremely slow processing in MoveNet
                num_people = min(len(det_results[0].boxes), max_detections)
                
                # Sort by confidence and take top N
                confidences = det_results[0].boxes.conf.cpu().numpy()
                top_indices = np.argsort(confidences)[::-1][:num_people]
                
                for idx in top_indices:
                    bbox = det_results[0].boxes.xyxy[idx].cpu().numpy()
                    conf = float(det_results[0].boxes.conf[idx].cpu().numpy())
                    x1, y1, x2, y2 = map(int, bbox)
                    candidate_boxes.append((x1, y1, x2, y2, conf))

        # ---------------------------------------------------------
        # PROCESS BOXES (Run MoveNet on Crops)
        # ---------------------------------------------------------
        detections = []
        
        for x1, y1, x2, y2, box_conf in candidate_boxes:
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Skip very small crops (likely false positives or bad GT)
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 10 or (y2-y1) < 10:
                continue
            
            person_crop = image[y1:y2, x1:x2]
            
            # Run pose estimation on this crop
            keypoints_crop = self._predict_single_crop(person_crop)
            
            if keypoints_crop is None:
                continue
            
            # Convert keypoints from crop coordinates to original image coordinates
            keypoints_original = keypoints_crop.copy()
            keypoints_original[:, 0] += x1  # Add x offset
            keypoints_original[:, 1] += y1  # Add y offset
            
            # Use average score of keypoints as pose confidence
            pose_score = keypoints_crop[:, 2].mean()
            
            # Final score is combination of box confidence and pose confidence
            final_score = box_conf * pose_score
            
            detections.append({
                'keypoints': keypoints_original,
                'score': float(final_score),
                'bbox': np.array([x1, y1, x2, y2])
            })
        
        return detections
    
    def _predict_single_crop(self, image):
        """
        Run MoveNet on a single cropped image
        Returns keypoints in crop coordinates
        """
        if image.size == 0: return None

        # Convert BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # Resize while preserving aspect ratio
        scale = self.input_size / max(orig_h, orig_w)
        resized_h = int(orig_h * scale)
        resized_w = int(orig_w * scale)

        # Handle degenerate resize cases
        if resized_h == 0 or resized_w == 0: return None

        resized = cv2.resize(image_rgb, (resized_w, resized_h))

        # Pad to square (MoveNet requirement)
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:resized_h, :resized_w] = resized

        # Prepare input tensor (batch size MUST be 1)
        input_tensor = tf.expand_dims(padded, axis=0)
        input_tensor = tf.cast(input_tensor, tf.int32)

        # Inference (single image only)
        outputs = self.movenet(input_tensor)
        keypoints = outputs["output_0"].numpy()[0, 0]  # (17, 3) -> (y, x, score)

        # Coordinate recovery
        keypoints_coco = np.zeros((17, 3), dtype=np.float32)

        for i in range(17):
            y_norm, x_norm, score = keypoints[i]

            # From normalized square -> padded image
            x_pad = x_norm * self.input_size
            y_pad = y_norm * self.input_size

            # Discard points that fall in padded area (outside the actual image content)
            if x_pad > resized_w or y_pad > resized_h:
                keypoints_coco[i] = [0.0, 0.0, 0.0]
                continue

            # Undo resize - convert back to crop coordinates
            x_crop = x_pad / scale
            y_crop = y_pad / scale

            keypoints_coco[i] = [x_crop, y_crop, score]

        return keypoints_coco
    
    def predict_single(self, image):
        """Single person prediction (highest confidence)"""
        all_detections = self.predict(image)
        if len(all_detections) == 0:
            return None
        return max(all_detections, key=lambda x: x['score'])
    
    def predict_batch(self, images):
        """Process multiple images"""
        predictions = []
        for img in images:
            predictions.append(self.predict(img))
        return predictions

if __name__ == "__main__":
    # Test logic
    estimator = MoveNetEstimator(variant='lightning')
    
    # Create dummy image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1) # Mock person
    
    # 1. Test Standard YOLO
    print("Testing Standard Mode...")
    res = estimator.predict(image)
    print(f"Standard detections: {len(res)}")
    
    # 2. Test GT Boxes
    print("Testing GT Mode...")
    # Mock COCO box [x, y, w, h]
    fake_gt = [[100, 100, 200, 200]] 
    res_gt = estimator.predict(image, gt_boxes=fake_gt)
    print(f"GT detections: {len(res_gt)}")