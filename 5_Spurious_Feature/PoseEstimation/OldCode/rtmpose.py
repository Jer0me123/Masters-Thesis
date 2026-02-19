import cv2
import numpy as np
import time
from pathlib import Path
import urllib.request
import zipfile
import tempfile
import shutil

from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown


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
        device: str = "cuda:0",
        yolo_weights: str = "yolov8n.pt",
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
        # cv2.destroyWindow(self._viz_win)


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
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    estimator = MMPoseTopDownEstimator(
        model_variant="w48_256",
        use_yolo=True,   # toggle here
    )

    image = cv2.imread(
        r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\Coco\val2017\000000000785.jpg" #000000001000.jpg" #
    )

    detections = estimator.predict(image)
    print(f"Detected {len(detections)} people")

    vis = image.copy()
    for person in detections:
        for x, y, c in person["keypoints"]:
            if c > 0.5:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        x1, y1, x2, y2 = person["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("Pose Estimation", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_human_pose_demo.md

# https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html#topdown-heatmap-hrnet-on-coco