import cv2
import numpy as np
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import urllib.request
import zipfile
import tempfile
import shutil
import torch

from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
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


# ============================================================
# HRNet MODEL REGISTRY (COCO TOP-DOWN)
# ============================================================

HRNET_MODELS = {
    "w32_256": {
        "config": "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
        ),
    },
    "w32_384": {
        "config": "td-hm_hrnet-w32_8xb64-210e_coco-384x288.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth"
        ),
    },
    "w48_256": {
        "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
        ),
    },
    "w48_384": {
        "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
            "td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth"
        ),
    },
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
# ENSURE MMPOSE CONFIGS
# ============================================================

def ensure_mmpose_configs(root: Path = Path("models/mmpose_configs")) -> Path:
    """
    Downloads the MMPose GitHub 'configs/' directory once and caches it locally.

    Returns:
        Path to local configs root, i.e. .../models/mmpose_configs/configs
    """
    configs_dir = root / "configs"
    if configs_dir.exists():
        return configs_dir

    print("[MMPose] Downloading config repository...")
    url = "https://github.com/open-mmlab/mmpose/archive/refs/heads/main.zip"

    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "mmpose.zip"
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # extract only configs
            for name in zf.namelist():
                if "mmpose-main/configs/" in name:
                    zf.extract(name, tmp)

        extracted = Path(tmp) / "mmpose-main" / "configs"
        shutil.copytree(extracted, configs_dir)

    print("[MMPose] Configs ready:", configs_dir)
    return configs_dir


# ============================================================
# TOP-DOWN POSE ESTIMATOR
# ============================================================

class MMPoseTopDownEstimator:
    """
    Evaluator-compatible MMPose top-down estimator.

    Key requirement:
      - inference_topdown expects XYXY boxes: [x1,y1,x2,y2]
      - COCO GT boxes are XYWH: [x,y,w,h]
    This wrapper converts GT boxes in ORACLE mode safely.
    """

    def __init__(
        self,
        model_variant: str = "w48_256",
        use_yolo: bool = True,
        yolo_weights: str = "models/yolov8m.pt",
    ):  
        if model_variant not in HRNET_MODELS:
            raise ValueError(
                f"Invalid model_variant: {model_variant}. "
                f"Choose from: {list(HRNET_MODELS.keys())}"
            )

        self.use_yolo = use_yolo

        configs_root = ensure_mmpose_configs()
        variant = HRNET_MODELS[model_variant]

        self.config_file = (
            configs_root
            / "body_2d_keypoint"
            / "topdown_heatmap"
            / "coco"
            / variant["config"]
        )

        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = init_model(
            str(self.config_file),
            variant["checkpoint"],  # MMPose will download/cache this automatically
            device=device,
        )

        self.model_name = f"HRNet-{model_variant.upper()}"
        self.detector = YOLO(yolo_weights) if use_yolo else None

        # Visualization window name
        self._viz_win = "Oracle Pose Debug"

    # ========================================================
    # MAIN API
    # ========================================================

    def predict(
        self,
        image: np.ndarray,
        gt_boxes: np.ndarray | None = None,
        conf_threshold: float = 0.25,
        visualize: bool = False,
        gt_format: str = "coco_xywh",
    ):
        """
        Args:
            image: BGR image (OpenCV).
            gt_boxes: optional GT boxes for ORACLE mode.
            conf_threshold: YOLO confidence threshold.
            visualize: if True, show debug visualization (ORACLE only by default).
            gt_format: one of {"coco_xywh", "xyxy"} when gt_boxes is provided.

        Returns:
            List of detections:
            [{
              "keypoints": (17,3) [x,y,score],
              "score": float,
              "bbox": (4,) XYXY
            }]
        """

        # ---------------- ORACLE MODE ----------------
        if gt_boxes is not None:
            boxes_xyxy = self._ensure_xyxy(gt_boxes, image=image, fmt=gt_format)
            detections = self._predict_from_boxes(image, boxes_xyxy, det_scores=None)

            if visualize:
                self._visualize_oracle(image, detections, boxes_xyxy)

            return detections

        # ---------------- FULL IMAGE (no detector) ----------------
        if not self.use_yolo:
            pose_results = inference_topdown(self.model, image)
            return self._format_full_image(image, pose_results)

        # ---------------- YOLO MODE ----------------
        det = self.detector(image, classes=[0], conf=conf_threshold, verbose=False)
        if len(det[0].boxes) == 0:
            return []

        boxes_xyxy = det[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        det_scores = det[0].boxes.conf.cpu().numpy().astype(np.float32)

        boxes_xyxy = self._clip_boxes_xyxy(boxes_xyxy, image)
        boxes_xyxy = self._filter_invalid_boxes(boxes_xyxy, det_scores=det_scores)[0]
        det_scores = self._filter_invalid_boxes(boxes_xyxy, det_scores=det_scores)[1]

        if boxes_xyxy.shape[0] == 0:
            return []

        return self._predict_from_boxes(image, boxes_xyxy, det_scores)

    def predict_single(self, image, conf_threshold=0.25):
        """
        Returns highest-confidence single-person detection
        """
        detections = self.predict(image, conf_threshold=conf_threshold)
        if len(detections) == 0:
            return None
        return max(detections, key=lambda x: x['score'])

    # ========================================================
    # BOX FORMAT / VALIDATION
    # ========================================================

    @staticmethod
    def _coco_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        """
        COCO: [x, y, w, h] -> [x1, y1, x2, y2]
        """
        b = np.asarray(boxes_xywh, dtype=np.float32)
        out = b.copy()
        out[:, 2] = b[:, 0] + b[:, 2]
        out[:, 3] = b[:, 1] + b[:, 3]
        return out

    @staticmethod
    def _xyxy_to_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
        return np.asarray(boxes_xyxy, dtype=np.float32)

    def _ensure_xyxy(self, boxes: np.ndarray, image: np.ndarray, fmt: str) -> np.ndarray:
        """
        Convert incoming GT boxes to XYXY in pixel space and clip to image.
        """
        boxes = np.asarray(boxes)

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        if boxes.shape[1] != 4:
            raise ValueError(f"Expected boxes Nx4, got shape={boxes.shape}")

        fmt = (fmt or "").lower().strip()
        if fmt in ("coco", "coco_xywh", "xywh"):
            boxes_xyxy = self._coco_xywh_to_xyxy(boxes)
        elif fmt in ("xyxy",):
            boxes_xyxy = self._xyxy_to_xyxy(boxes)
        else:
            raise ValueError(f"Unknown gt_format='{fmt}'. Use 'coco_xywh' or 'xyxy'.")

        boxes_xyxy = self._clip_boxes_xyxy(boxes_xyxy, image)
        boxes_xyxy, _ = self._filter_invalid_boxes(boxes_xyxy, det_scores=None)
        return boxes_xyxy

    @staticmethod
    def _clip_boxes_xyxy(boxes_xyxy: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Clip XYXY boxes to image bounds.
        """
        h, w = image.shape[:2]
        b = boxes_xyxy.copy()
        b[:, 0] = np.clip(b[:, 0], 0, w - 1)
        b[:, 2] = np.clip(b[:, 2], 0, w - 1)
        b[:, 1] = np.clip(b[:, 1], 0, h - 1)
        b[:, 3] = np.clip(b[:, 3], 0, h - 1)
        return b

    @staticmethod
    def _filter_invalid_boxes(boxes_xyxy: np.ndarray, det_scores: np.ndarray | None):
        """
        Remove boxes where x2<=x1 or y2<=y1 after clipping.
        """
        if boxes_xyxy.shape[0] == 0:
            return boxes_xyxy, det_scores

        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        keep = (x2 > x1 + 1.0) & (y2 > y1 + 1.0)

        boxes_out = boxes_xyxy[keep]
        if det_scores is None:
            return boxes_out, None
        return boxes_out, det_scores[keep]

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _predict_from_boxes(self, image: np.ndarray, boxes_xyxy: np.ndarray, det_scores: np.ndarray | None):
        """
        Boxes MUST be XYXY here.
        """
        # Defensive checks to prevent returning to the old failure mode
        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(f"Expected boxes_xyxy Nx4, got shape={boxes_xyxy.shape}")

        # MMPose expects float array; avoid int/uint issues
        boxes_xyxy = boxes_xyxy.astype(np.float32, copy=False)

        pose_results = inference_topdown(self.model, image, bboxes=boxes_xyxy)

        detections = []
        for i, pose in enumerate(pose_results):
            kpts = pose.pred_instances.keypoints[0]            # (17,2)
            kpt_scores = pose.pred_instances.keypoint_scores[0]  # (17,)

            detections.append({
                "keypoints": np.concatenate([kpts, kpt_scores[:, None]], axis=1),  # (17,3)
                "score": float(det_scores[i]) if det_scores is not None else float(kpt_scores.mean()),
                "bbox": boxes_xyxy[i],
            })

        return detections

    @staticmethod
    def _format_full_image(image: np.ndarray, pose_results):
        detections = []
        for pose in pose_results:
            kpts = pose.pred_instances.keypoints[0]
            kpt_scores = pose.pred_instances.keypoint_scores[0]

            detections.append({
                "keypoints": np.concatenate([kpts, kpt_scores[:, None]], axis=1),
                "score": float(kpt_scores.mean()),
                "bbox": np.array([0, 0, image.shape[1], image.shape[0]], dtype=np.float32),
            })
        return detections

    # ========================================================
    # VISUALIZATION (ORACLE DEBUG)
    # ========================================================

    def _visualize_oracle(self, image: np.ndarray, detections, boxes_xyxy: np.ndarray):
        """
        Visualize GT boxes (blue) and predicted keypoints (green).
        """
        vis = image.copy()

        # Draw boxes
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw predicted keypoints
        for det in detections:
            for x, y, c in det["keypoints"]:
                if float(c) > 0.3:
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.putText(
            vis,
            f"{self.model_name} | ORACLE MODE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        cv2.imshow(self._viz_win, vis)
        cv2.waitKey(0)

    # ========================================================
    # OPTIONAL UTILITIES
    # ========================================================

    def benchmark(self, image: np.ndarray, runs: int = 50):
        for _ in range(10):
            self.predict(image, visualize=False)

        t0 = time.time()
        for _ in range(runs):
            self.predict(image, visualize=False)
        dt = (time.time() - t0) / runs

        return {
            "model": self.model_name,
            "avg_time_ms": dt * 1000,
            "fps": 1.0 / dt,
        }


# ============================================================
# DIRECTORY PIPELINE
# ============================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_variant: str,
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
    estimator = MMPoseTopDownEstimator(
        model_variant=model_variant,
        use_yolo=True,
        yolo_weights=yolo_model,
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


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser("MMPose HRNet + YOLO Pose Directory Pipeline")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-variant", default="w48_256", 
                    choices=list(HRNET_MODELS.keys()))
    ap.add_argument("--yolo-model", default="models\yolov8m.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    process_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        args.model_variant,
        args.yolo_model,
        args.batch_size,
        args.num_workers,
        args.conf,
        args.draw,
    )

# .venv_test\Scripts\python.exe rtmpose.py --input-dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned\Female_Accountant" --output-dir "test_rtmpose" --draw

# https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_human_pose_demo.md
# https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html#topdown-heatmap-hrnet-on-coco
