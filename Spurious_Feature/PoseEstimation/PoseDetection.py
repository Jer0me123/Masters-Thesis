"""
================================================================================
POSE FEATURE EXTRACTION PIPELINE (YOLOv8 → COCO-ALIGNED POSE VECTORS)
================================================================================

PURPOSE
-------
This script implements a pose-only feature extraction pipeline aligned with the
experimental protocol used in the paper "Gender Artifacts in Visual Datasets".

The goal of this pipeline is NOT to perform pose estimation for downstream
applications, but to convert raw images into low-dimensional, geometry-only
representations of human pose that can be used as input to a low-capacity
Multi-Layer Perceptron (MLP) probe.

By discarding all pixel-level appearance information and retaining only pose
geometry, the pipeline enables controlled experiments that measure whether
spurious correlations (e.g., gender or skin tone) can be inferred from pose alone.

This script therefore serves as a measurement instrument, not a predictive model.

--------------------------------------------------------------------------------
HIGH-LEVEL PIPELINE OVERVIEW
--------------------------------------------------------------------------------

For each image in a directory tree, the script performs the following steps:

1. Recursively scans the input directory for image files
2. Loads images efficiently using a parallel image loader
3. Runs YOLOv8-Pose to detect people and COCO-17 keypoints
4. Enforces a single-person constraint per image
5. Converts YOLO keypoints to COCO format
6. Applies COCO-style visibility masking
7. Normalizes pose geometry to a fixed area
8. Serializes pose vectors to JSONL
9. Optionally renders pose visualizations for debugging

Each step is designed to align exactly with the assumptions made in the paper.

--------------------------------------------------------------------------------
CONFIGURATION CONSTANTS
--------------------------------------------------------------------------------

TARGET_POSE_AREA = 4000.0

This value is taken directly from the paper. After centering each pose, the
bounding box area of the visible joints is scaled to exactly 4000 pixels.

This normalization removes:
- Absolute scale (distance to camera)
- Person height differences
- Dataset-specific zoom artifacts

As a result, the downstream MLP sees pose geometry that is invariant to scale.
Changing this value would break comparability with the paper.

IMG_EXTS

Restricts the pipeline to standard image formats. This avoids accidentally
processing non-image files (e.g., masks, metadata, thumbnails) and reduces
unnecessary I/O overhead.

COCO_KEYPOINT_NAMES

Defines the canonical COCO-17 joint ordering:

    ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
     "left_wrist", "right_wrist", "left_hip", "right_hip",
     "left_knee", "right_knee", "left_ankle", "right_ankle"]

This ordering is critical. All saved keypoint vectors strictly follow this order,
matching the COCO annotation format and the pose representation assumed in the
paper and in the MLP probe.

COCO_SKELETON

Defines joint connectivity for visualization only. Skeleton edges are never used
as model inputs and do not affect the saved pose vectors. They exist purely for
sanity-checking and debugging.

--------------------------------------------------------------------------------
FAST IMAGE LOADER
--------------------------------------------------------------------------------

The FastImageLoader class decouples disk I/O from GPU inference.

Key design choices:
- Uses a ThreadPoolExecutor to load images in parallel
- Attempts TurboJPEG decoding first (faster for JPEG-heavy datasets such as LAION)
- Falls back to OpenCV for robustness

This ensures:
- The GPU is not idle waiting for disk reads
- The pipeline scales to large datasets
- Corrupt images do not crash the run

This component is purely engineering infrastructure and does not affect the
scientific methodology.

--------------------------------------------------------------------------------
YOLOv8 POSE ESTIMATOR
--------------------------------------------------------------------------------

YOLOv8-Pose is used because:
- It outputs COCO-17 keypoints
- It is fast enough for large-scale extraction
- It matches the pose representation assumed in the paper

SINGLE-PERSON ENFORCEMENT

Although YOLO may detect multiple people in an image, this pipeline enforces
exactly one person per image by selecting the detection with the highest box
confidence score.

This is implemented by:

    best = argmax(box_confidence)

All other detected people are discarded.

If no person is detected, the image is encoded as an all-zero pose vector.

This design choice is intentional and paper-aligned:
- The MLP expects exactly one fixed-length vector per image
- The paper does not aggregate multiple poses
- Background people are ignored rather than averaged or filtered

--------------------------------------------------------------------------------
COCO CONVERSION AND VISIBILITY MASKING
--------------------------------------------------------------------------------

YOLO outputs keypoints as (x, y, confidence). COCO expects (x, y, visibility).

Visibility semantics:
- visibility = 2 → keypoint is visible and reliable
- visibility = 0 → keypoint is not labeled / not reliable

For each joint:
- If confidence >= kp_conf_thr:
    (x, y, v) = (x, y, 2)
- Else:
    (x, y, v) = (0, 0, 0)

This exactly matches COCO annotation conventions.

Visibility masking is essential to:
- Remove noise from low-confidence detections
- Prevent the MLP from learning from unreliable joints
- Ensure missing joints behave identically to COCO ground truth

--------------------------------------------------------------------------------
POSE NORMALIZATION
--------------------------------------------------------------------------------

Two normalized representations are produced:
- normalized_keypoints
- normalized_keypoints_with_visibility

Normalization steps:
1. Consider only visible joints
2. Center the pose by subtracting the mean joint position
3. Compute the bounding box area of visible joints
4. Scale the pose so that area == TARGET_POSE_AREA (4000)

Invisible joints remain at (0, 0) after normalization.

This removes:
- Absolute image position
- Person scale
- Camera distance cues

Only relative joint geometry remains, which is the intended signal for the MLP.

--------------------------------------------------------------------------------
SAVED OUTPUT FORMAT (JSONL)
--------------------------------------------------------------------------------

Each image produces exactly one JSON record with the following fields:

- keypoints:
    Flat list of length 34: [x0, y0, x1, y1, ..., x16, y16]

- keypoints_with_visibility:
    Flat list of length 51: [x0, y0, v0, ..., x16, y16, v16]

- normalized_keypoints:
    Centered + area-normalized version of keypoints

- normalized_keypoints_with_visibility:
    Normalized pose with COCO visibility preserved

All lists follow strict COCO joint ordering.

This flat representation:
- Matches COCO annotations exactly
- Is directly compatible with NumPy and PyTorch
- Is suitable for low-capacity MLP probes

--------------------------------------------------------------------------------
RESUME LOGIC
--------------------------------------------------------------------------------

The pipeline supports safe resumption by tracking processed image filenames
already written to poses.jsonl.

This enables:
- Crash-safe long-running jobs
- Incremental dataset construction
- Efficient experimentation without recomputation

Resume logic does not affect pose values or experimental validity.

--------------------------------------------------------------------------------
CLI ARGUMENTS
--------------------------------------------------------------------------------

--model-size:
    Controls YOLOv8 backbone size (speed vs accuracy trade-off)

--conf:
    Detection confidence threshold for person boxes

--kp-conf-thr:
    Confidence threshold for individual keypoints (COCO visibility masking)

--batch-size:
    Controls GPU memory usage and throughput

--num-workers:
    Controls parallel image loading throughput

--draw:
    Enables optional visualization of poses for debugging only

None of these parameters are tuned for performance. They control extraction
quality and computational feasibility, not learning behavior.

--------------------------------------------------------------------------------
RELATIONSHIP TO THE MLP PROBE
--------------------------------------------------------------------------------

This script produces the exact numeric pose representation consumed by the MLP
probe used in the paper.

Together, the two scripts form a complete probing pipeline:

    Image → Pose → Normalized Geometry → Low-Capacity MLP → Bias Measurement

IMPORTANT: Yolov8-l was chosen for pose extraction as it acheives a good balance between speed and accuracy, 
eventhough the rtmpose models were more accurate their speed was lackluster as can be seen in the model_comparison.png file
or by running the model_evaluation_comparison.py file.


All design choices ensure:
- Minimal representational capacity
- No appearance information
- Full reproducibility
- Faithful alignment with the paper

================================================================================
"""

"""
================================================================================
POSE FEATURE EXTRACTION PIPELINE (YOLOv8 → COCO-ALIGNED POSE VECTORS)
================================================================================
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

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

        model_name = f"models/yolov8{model_size}-pose.pt"
        print(f"[YOLO] Loading {model_name} on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        self.model.fuse()

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
            cv2.line(
                img,
                (int(kps[i,0]),int(kps[i,1])),
                (int(kps[j,0]),int(kps[j,1])),
                (0,255,255),2
            )

# ============================================================
# PIPELINE
# ============================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_size: str,
    batch_size: int,
    num_workers: int,
    conf: float,
    kp_conf_thr: float,
    draw_flag: bool,
    resize: Optional[Tuple[int,int]],
    exclude_dirs: List[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "poses.jsonl"

    processed = set()
    if jsonl_path.exists():
        for line in open(jsonl_path,"r",encoding="utf-8"):
            processed.add(json.loads(line)["image"])

    exclude_dirs = {d.lower() for d in exclude_dirs}

    print("Resuming, found", len(processed), "already processed images")
    print("Scanning image directory prior to processing")

    paths = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if p.name in processed:
            continue
        if any(part.lower() in exclude_dirs for part in p.parts):
            continue
        paths.append(p)

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

                if draw_flag and d is not None and img is not None:
                    vis = img.copy()
                    draw_pose(vis, xyv)

                    if resize is not None:
                        vis = cv2.resize(vis, resize, interpolation=cv2.INTER_AREA)

                    out = output_dir / p.relative_to(input_dir)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), vis)

    loader.shutdown()

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser("YOLOv8 → COCO Pose JSON")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_size", default="s", choices=["n","s","m","l","x"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp-conf-thr", type=float, default=0.5)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--resize", nargs=2, type=int, metavar=("W","H"), help="Resize drawn output images to (W H)")
    ap.add_argument("--exclude_dirs", nargs="*", default=["facemesh"], help="Directory names to exclude during recursive scan")

    args = ap.parse_args()

    process_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        args.model_size,
        args.batch_size,
        args.num_workers,
        args.conf,
        args.kp_conf_thr,
        args.draw,
        tuple(args.resize) if args.resize else None,
        args.exclude_dirs,
    )

# ============================================================
# EXAMPLE USAGE
# ============================================================
# python PoseDetection.py --image_dir "path/to/images" --batch_size 16 --num_workers 8 ---output-dir "path/to/images" --exclude_dirs facemesh --model_size l --conf 0.25 --kp-conf-thr 0.5

# python PoseDetection.py --image_dir "path/to/images" --batch_size 16 --num_workers 8 ---output-dir "path/to/images" --exclude_dirs facemesh --model_size l --conf 0.25 --kp-conf-thr 0.5 --draw --resize 224 224

# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --model_size l -> Yolo was the best performing model overall in terms of speed and accuracy, with the l variant achieving the best balance
# --conf 0.25 -> Controls the minimum confidence required for a person bounding box to be considered a valid detection.
# --kp-conf-thr 0.5 -> Controls whether an individual keypoint is considered visible.
# --draw -> This signals to the program to draw the pose points onto the image and store the results
# --resize 224 224 -> Only works with --draw. This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.


# python yolov8_pose_final.py --input-dir "E:\ImageRetrieval\Professions_125k_Cleaned\Female_Actuarial_Analyst" --output-dir "E:\Test" --model-size l --device cuda --batch-size 16 --num-workers 8 --conf 0.25

# .venv_test\Scripts\python.exe yolov8_pose_final.py --input-dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output-dir "E:\Test" --model-size l --device cuda --batch-size 16 --num-workers 8 --conf 0.25 --draw


# .venv_test\Scripts\python.exe PoseDetection.py --input_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir "E:\Test3" --model_size l --device cuda --batch_size 16 --num_workers 8 --conf 0.25


# python PoseDetection.py --input_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir "E:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\PoseDetection" --model_size l --exclude_dirs face_crops --batch_size 16 --num_workers 8 --conf 0.25 --kp-conf-thr 0.5 --draw --resize 224 224

# python PoseDetection.py --input_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\PoseDetection" --model_size l --exclude_dirs facemesh --batch_size 16 --num_workers 8 --conf 0.25 --kp-conf-thr 0.5 --draw --resize 224 224