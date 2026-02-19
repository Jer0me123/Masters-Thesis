import os
import cv2
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch

# ============================================================
# CONFIG
# ============================================================

TARGET_POSE_AREA = 4000.0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

COCO_KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

COCO_SKELETON = [
    (15,13),(13,11),(16,14),(14,12),
    (11,12),(5,11),(6,12),
    (5,6),(5,7),(6,8),
    (7,9),(8,10),
    (1,2),(0,1),(0,2),(1,3),(2,4)
]

# ============================================================
# FAST IMAGE LOADER
# ============================================================

class FastImageLoader:
    def __init__(self, num_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            from turbojpeg import TurboJPEG
            self.jpeg_decoder = TurboJPEG(
                r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll"
            )
            self.use_turbo = True
            print("[ImageLoader] Using TurboJPEG")
        except Exception:
            self.use_turbo = False
            print("[ImageLoader] TurboJPEG not available, using cv2")

    def load_image(self, path: Path):
        try:
            if self.use_turbo and path.suffix.lower() in {".jpg", ".jpeg"}:
                try:
                    with open(path, "rb") as f:
                        return self.jpeg_decoder.decode(f.read())
                except Exception:
                    pass
            return cv2.imread(str(path))
        except Exception:
            return None

    def load_batch(self, paths: List[Path]):
        return list(self.executor.map(self.load_image, paths))

    def shutdown(self):
        self.executor.shutdown(wait=False)

# ============================================================
# YOLOv8 POSE
# ============================================================

class YOLOv8PoseEstimator:
    def __init__(self, model_size: str):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = f"models\yolov8{model_size}-pose.pt"
        print(f"[YOLO] Loading {model_name} on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        self.model.fuse()

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

    def predict_batch(self, images, conf):
        results = self.model(images, verbose=False, conf=conf)
        out = []
        for r in results:
            if r.keypoints is None or len(r.boxes) == 0:
                out.append(None)
            else:
                best = np.argmax(r.boxes.conf.cpu().numpy())
                out.append({
                    "keypoints": r.keypoints.data[best].cpu().numpy(),  # (17,3)
                    "bbox": r.boxes.xyxy[best].cpu().numpy(),
                    "score": float(r.boxes.conf[best].cpu().numpy())
                })
        return out

# ============================================================
# COCO CONVERSION + NORMALIZATION
# ============================================================

def to_coco_lists(kps, kp_conf_thr):
    """
    YOLO (17,3) -> COCO flat lists
    """
    xy = []
    xyv = []

    for x, y, c in kps:
        if c >= kp_conf_thr:
            xy.extend([float(x), float(y)])
            xyv.extend([float(x), float(y), 2])
        else:
            xy.extend([0.0, 0.0])
            xyv.extend([0.0, 0.0, 0])

    return xy, xyv


def normalize_xy(flat_xy):
    xy = np.array(flat_xy, dtype=np.float32).reshape(17, 2)

    mask = ~(xy == 0).all(axis=1)
    if not mask.any():
        return [0.0] * 34

    visible = xy[mask]
    center = visible.mean(axis=0)
    xy[mask] -= center

    min_xy = visible.min(axis=0)
    max_xy = visible.max(axis=0)
    area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
    scale = np.sqrt(TARGET_POSE_AREA / max(area, 1e-6))

    xy[mask] *= scale
    return xy.flatten().tolist()


def normalize_xyv(flat_xyv):
    xyv = np.array(flat_xyv, dtype=np.float32).reshape(17, 3)
    xy = xyv[:, :2]
    v = xyv[:, 2]

    mask = v > 0
    if not mask.any():
        return [0.0] * 51

    visible = xy[mask]
    center = visible.mean(axis=0)
    xy[mask] -= center

    min_xy = visible.min(axis=0)
    max_xy = visible.max(axis=0)
    area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
    scale = np.sqrt(TARGET_POSE_AREA / max(area, 1e-6))

    xy[mask] *= scale
    xyv[:, :2] = xy
    return xyv.flatten().tolist()

# ============================================================
# DRAWING
# ============================================================

def draw_pose(img, kps_xyv):
    kps = np.array(kps_xyv).reshape(17,3)
    for x,y,v in kps:
        if v > 0:
            cv2.circle(img,(int(x),int(y)),3,(0,0,255),-1)
    for i,j in COCO_SKELETON:
        if kps[i,2]>0 and kps[j,2]>0:
            cv2.line(img,
                     (int(kps[i,0]),int(kps[i,1])),
                     (int(kps[j,0]),int(kps[j,1])),
                     (0,255,255),2)

# ============================================================
# PIPELINE
# ============================================================

def process_directory(
    input_dir, output_dir, model_size, 
    batch_size, num_workers, conf, kp_conf_thr,
    draw_flag
):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "poses.jsonl"

    processed = set()
    if jsonl_path.exists():
        for line in open(jsonl_path,"r",encoding="utf-8"):
            processed.add(json.loads(line)["image"])

    print("Resuming, found", len(processed), "already processed images")
    print("Scanning image directory prior to processing")
    paths = [
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
        and "facemesh" not in (x.lower() for x in p.parts)
        and p.name not in processed
    ]

    loader = FastImageLoader(num_workers)
    model = YOLOv8PoseEstimator(model_size)

    for i in tqdm(range(0,len(paths),batch_size)):
        batch = paths[i:i+batch_size]
        imgs = loader.load_batch(batch)

        preds = model.predict_batch(imgs, conf)

        with open(jsonl_path,"a",encoding="utf-8") as f:
            for p,img,d in zip(batch,imgs,preds):
                if d is None:
                    xy = [0.0]*34
                    xyv = [0.0]*51
                else:
                    xy, xyv = to_coco_lists(d["keypoints"], kp_conf_thr)

                rec = {
                    "parent_dir": p.parent.name,
                    "image": p.name,
                    "keypoints": xy,
                    "keypoints_with_visibility": xyv,
                    "normalized_keypoints": normalize_xy(xy),
                    "normalized_keypoints_with_visibility": normalize_xyv(xyv)
                }

                f.write(json.dumps(rec) + "\n")

                if draw_flag and d is not None:
                    vis = img.copy()
                    draw_pose(vis, xyv)
                    out = output_dir / p.relative_to(input_dir)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), vis)

    loader.shutdown()

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser("YOLOv8 → COCO Pose JSON")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-size", default="s", choices=["n","s","m","l","x"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp-conf-thr", type=float, default=0.5)
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    process_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        args.model_size,
        args.batch_size,
        args.num_workers,
        args.conf,
        args.kp_conf_thr,
        args.draw
    )

# .venv_test\Scripts\python.exe yolov8_pose.py --input-dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned\Female_Accountant" --output-dir "test_yolo" --draw