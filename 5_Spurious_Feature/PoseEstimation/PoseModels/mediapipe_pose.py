import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import urllib.request
from ultralytics import YOLO
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

MEDIAPIPE_MODELS = {
    0: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    1: (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    2: (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
}


# ============================================================
# FAST IMAGE LOADER
# ============================================================

class FastImageLoader:
    def __init__(self, num_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def load(self, path: Path):
        try:
            return cv2.imread(str(path))
        except Exception:
            return None

    def load_batch(self, paths: List[Path]):
        return list(self.executor.map(self.load, paths))

    def shutdown(self):
        self.executor.shutdown(wait=False)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_mediapipe_model(complexity: int, model_dir="models/mediapipe") -> Path:
    model_name, url = MEDIAPIPE_MODELS[complexity]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    if not model_path.exists():
        print(f"[MediaPipe] Downloading {model_name} ...")
        urllib.request.urlretrieve(url, model_path)

    return model_path


# ============================================================
# MEDIAPIPE POSE ESTIMATOR
# ============================================================

class MediaPipePoseEstimator:
    # Changed default complexity to 2 (Heavy) for better accuracy
    def __init__(self, complexity=2, yolo_model='models/yolov8m.pt'):
        
        model_path = ensure_mediapipe_model(complexity)
        self.model_name = f"MediaPipe-Pose-C{complexity}"

        # Relaxed thresholds to trust YOLO more and allow MP to find difficult poses
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1, 
            min_pose_detection_confidence=0.1, # Lowered from 0.5
            min_pose_presence_confidence=0.1,  # Lowered from 0.5
            min_tracking_confidence=0.4
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        # Upgraded to YOLOv8m (Medium) for better detection recall
        print(f"[Info] Loading {yolo_model} for person detection...")
        self.detector = YOLO(yolo_model)

        self.mp_to_coco = {
            0: 0,    # nose
            2: 1,    # left_eye
            5: 2,    # right_eye
            7: 3,    # left_ear
            8: 4,    # right_ear
            11: 5,   # left_shoulder
            12: 6,   # right_shoulder
            13: 7,   # left_elbow
            14: 8,   # right_elbow
            15: 9,   # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }

    def predict(self, image, gt_boxes=None, conf_threshold=0.25, max_detections=100):
        h_img, w_img = image.shape[:2]
        detections = []
        
        candidate_boxes = []

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
            yolo_results = self.detector(image, classes=[0], verbose=False, 
                                       conf=conf_threshold, iou=0.5)
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes
                sorted_indices = np.argsort(boxes.conf.cpu().numpy())[::-1][:max_detections]
                
                for idx in sorted_indices:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    candidate_boxes.append((x1, y1, x2, y2, conf))

        # ---------------------------------------------------------
        # COMMON: Process whatever boxes we found (GT or YOLO)
        # ---------------------------------------------------------
        for x1, y1, x2, y2, box_conf in candidate_boxes:
            
            # 1. SQUARE PADDING (Crucial for MediaPipe)
            w_box, h_box = x2 - x1, y2 - y1
            cx, cy = x1 + w_box // 2, y1 + h_box // 2
            
            # 1.5x padding to include limbs
            size = int(max(w_box, h_box) * 1.5)
            
            x1_sq = max(0, cx - size // 2)
            y1_sq = max(0, cy - size // 2)
            x2_sq = min(w_img, cx + size // 2)
            y2_sq = min(h_img, cy + size // 2)
            
            crop = image[y1_sq:y2_sq, x1_sq:x2_sq]
            if crop.size == 0: continue
            
            # 2. Run MediaPipe
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = np.ascontiguousarray(crop_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            
            try:
                result = self.landmarker.detect(mp_image)
            except Exception:
                continue

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            coco_keypoints = np.zeros((17, 3), dtype=np.float32)
            kp_confidences = []
            
            crop_h, crop_w = crop.shape[:2]
            
            for mp_idx, coco_idx in self.mp_to_coco.items():
                lm = landmarks[mp_idx]
                global_x = x1_sq + (lm.x * crop_w)
                global_y = y1_sq + (lm.y * crop_h)
                coco_keypoints[coco_idx] = [global_x, global_y, lm.visibility]
                kp_confidences.append(lm.visibility)

            # 3. Robust Scoring (Top 5 Keypoints)
            pose_score = np.mean(sorted(kp_confidences)[-5:])
            
            # If using GT, box_conf is 1.0, so score relies purely on Pose Confidence
            final_score = box_conf * pose_score

            detections.append({
                "keypoints": coco_keypoints,
                "score": float(final_score),
                "bbox": np.array([x1, y1, x2, y2])
            })

        return detections

    def predict_single(self, image, conf_threshold=0.25):
        all_detections = self.predict(image, conf_threshold=conf_threshold)
        if len(all_detections) == 0:
            return None
        return max(all_detections, key=lambda x: x['score'])

    def predict_batch(self, images):
        return [self.predict(img) for img in images]

    def close(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def __del__(self):
        self.close()


# ============================================================
# DIRECTORY PIPELINE
# ============================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
    complexity: int,
    yolo_model: str,
    batch_size: int,
    num_workers: int,
    conf: float,
    draw: bool,
):

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "poses.jsonl"

    processed = set()
    if jsonl_path.exists():
        for line in open(jsonl_path, "r", encoding="utf-8"):
            processed.add(json.loads(line)["image"])

    paths = [
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
        and "facemesh" not in (x.lower() for x in p.parts)
        and p.name not in processed
    ]

    print(f"Found {len(paths)} images to process")

    loader = FastImageLoader(num_workers)
    estimator = MediaPipePoseEstimator(
        complexity=complexity,
        yolo_model=yolo_model,
    )

    for i in tqdm(range(0, len(paths), batch_size)):
        batch = paths[i:i + batch_size]
        images = loader.load_batch(batch)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            for p, img in zip(batch, images):
                if img is None:
                    continue

                det = estimator.predict_single(img, conf_threshold=conf)
                if det is None:
                    record = {
                        "image": p.name,
                        "keypoints": [0.0] * 51,
                        "score": 0.0,
                    }
                else:
                    record = {
                        "image": p.name,
                        "keypoints": det["keypoints"].flatten().tolist(),
                        "score": det["score"],
                    }

                f.write(json.dumps(record) + "\n")

                if draw and det is not None:
                    vis = img.copy()
                    for x, y, v in det["keypoints"]:
                        if v > 0:
                            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    out = output_dir / p.relative_to(input_dir)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), vis)

    loader.shutdown()
    estimator.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser("MediaPipe Pose + YOLO Directory Pipeline")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="0=lite, 1=full, 2=heavy")
    ap.add_argument("--yolo-model", default="models/yolov8m.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    process_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        args.complexity,
        args.yolo_model,
        args.batch_size,
        args.num_workers,
        args.conf,
        args.draw,
    )

# .venv_test\Scripts\python.exe mediapipe_pose.py --input-dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned\Female_Accountant" --output-dir "test" --draw