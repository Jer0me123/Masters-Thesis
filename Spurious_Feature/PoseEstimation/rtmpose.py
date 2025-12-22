# # # # import cv2
# # # # import numpy as np
# # # # from mmpose.apis.inference import init_model, inference_topdown
# # # # import time
# # # # from ultralytics import YOLO

# # # # class RTMPoseEstimator:
# # # #     def __init__(self, model_size='m'):
# # # #         """
# # # #         model_size: 't' (tiny), 's', 'm', 'l'
# # # #         """
# # # #         # Configuration and checkpoint paths
# # # #         config_map = {
# # # #             't': 'rtmpose-t_8xb256-420e_coco-256x192.py',
# # # #             's': 'rtmpose-s_8xb256-420e_coco-256x192.py',
# # # #             'm': 'rtmpose-m_8xb256-420e_coco-256x192.py',
# # # #             'l': 'rtmpose-l_8xb256-420e_coco-384x288.py'
# # # #         }
        
# # # #         checkpoint_map = {
# # # #             't': 'rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # # #             's': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # # #             'm': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # # #             'l': 'rtmpose-l_simcc-coco_pt-aic-coco_420e-384x288.pth'
# # # #         }
        
# # # #         config_file = f'mmpose/configs/body_2d_keypoint/rtmpose/coco/{config_map[model_size]}'
# # # #         checkpoint_file = f'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/{checkpoint_map[model_size]}'
        
# # # #         self.model = init_model(config_file, checkpoint_file, device='cuda:0')
# # # #         self.model_name = f'RTMPose-{model_size.upper()}'
        
# # # #         # Person detector
# # # #         self.detector = YOLO('yolov8n.pt')
        
# # # #     def predict(self, image, conf_threshold=0.25):
# # # #         """
# # # #         Returns COCO format keypoints for ALL detected persons
# # # #         """
# # # #         # Detect all people
# # # #         det_results = self.detector(image, classes=[0], verbose=False, conf=conf_threshold)
        
# # # #         if len(det_results[0].boxes) == 0:
# # # #             return []
        
# # # #         # Get all bboxes
# # # #         bboxes = []
# # # #         person_confs = []
# # # #         for i in range(len(det_results[0].boxes)):
# # # #             bbox = det_results[0].boxes.xyxy[i].cpu().numpy()
# # # #             person_conf = det_results[0].boxes.conf[i].cpu().numpy()
# # # #             bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 1.0])
# # # #             person_confs.append(person_conf)
        
# # # #         # Convert to numpy array for MMPose
# # # #         bboxes_array = np.array(bboxes)
        
# # # #         # Run pose estimation on all detected people at once
# # # #         pose_results = inference_topdown(self.model, image, bboxes=bboxes_array)
        
# # # #         detections = []
# # # #         for i, pose_result in enumerate(pose_results):
# # # #             # Extract keypoints
# # # #             keypoints = pose_result.pred_instances.keypoints[0]  # [17, 2]
# # # #             scores = pose_result.pred_instances.keypoint_scores[0]  # [17]
            
# # # #             # Combine to COCO format [17, 3]
# # # #             keypoints_with_scores = np.concatenate([
# # # #                 keypoints, scores.reshape(-1, 1)
# # # #             ], axis=1)
            
# # # #             detections.append({
# # # #                 'keypoints': keypoints_with_scores,
# # # #                 'score': float(person_confs[i]),  # Use detection confidence
# # # #                 'bbox': det_results[0].boxes.xyxy[i].cpu().numpy()
# # # #             })
        
# # # #         return detections
    
# # # #     def predict_single(self, image):
# # # #         """Single person prediction (highest confidence)"""
# # # #         all_detections = self.predict(image)
# # # #         if len(all_detections) == 0:
# # # #             return None
# # # #         return max(all_detections, key=lambda x: x['score'])
    
# # # #     def predict_batch(self, images):
# # # #         """Process multiple images"""
# # # #         predictions = []
# # # #         for img in images:
# # # #             predictions.append(self.predict(img))
# # # #         return predictions
    
# # # #     def benchmark(self, image, num_runs=100):
# # # #         """Measure inference speed"""
# # # #         # Warmup
# # # #         for _ in range(10):
# # # #             self.predict(image)
        
# # # #         start = time.time()
# # # #         for _ in range(num_runs):
# # # #             self.predict(image)
# # # #         end = time.time()
        
# # # #         avg_time = (end - start) / num_runs
# # # #         fps = 1.0 / avg_time
        
# # # #         return {
# # # #             'model': self.model_name,
# # # #             'avg_time_ms': avg_time * 1000,
# # # #             'fps': fps
# # # #         }


# # # # # Example usage
# # # # if __name__ == "__main__":
# # # #     estimator = RTMPoseEstimator(model_size='s')
    
# # # #     # image = cv2.imread('test_image.jpg')
# # # #     image = cv2.imread(r'G:\Thesis\ImageRetrieval\Professions_125k_test\Female_Accountant\0.210_0000_1042168.jpg')
# # # #     all_people = estimator.predict(image)
    
# # # #     print(f"Detected {len(all_people)} people")
# # # #     for i, person in enumerate(all_people):
# # # #         print(f"  Person {i+1}: confidence {person['score']:.3f}")
    
# # # #     # Single person
# # # #     best_person = estimator.predict_single(image)
# # # #     if best_person:
# # # #         print(f"\nBest person: confidence {best_person['score']:.3f}")
    
# # # #     # Visualize all
# # # #     annotated = image.copy()
# # # #     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
# # # #     for idx, person in enumerate(all_people):
# # # #         color = colors[idx % len(colors)]
# # # #         for kp in person['keypoints']:
# # # #             x, y, conf = kp
# # # #             if conf > 0.5:
# # # #                 cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
        
# # # #         x1, y1, x2, y2 = person['bbox']
# # # #         cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
# # # #     cv2.imshow('RTMPose All People', annotated)
# # # #     cv2.waitKey(0)
    
# # # #     benchmark_results = estimator.benchmark(image)
# # # #     print(f"\nBenchmark Results:")
# # # #     print(f"Model: {benchmark_results['model']}")
# # # #     print(f"Average Time: {benchmark_results['avg_time_ms']:.2f} ms")
# # # #     print(f"FPS: {benchmark_results['fps']:.2f}")


# # # import cv2
# # # import numpy as np
# # # import time
# # # from pathlib import Path

# # # import mmpose
# # # from mmpose.apis.inference import init_model, inference_topdown
# # # from ultralytics import YOLO

# # # RTMPOSE_CONFIGS = {
# # #     "t": (
# # #         "rtmpose-t_8xb256-420e_coco-256x192.py",
# # #         "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
# # #         "configs/body_2d_keypoint/rtmpose/coco/"
# # #         "rtmpose-t_8xb256-420e_coco-256x192.py",
# # #     ),
# # #     "s": (
# # #         "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
# # #         # "rtmpose-s_8xb256-420e_coco-256x192.py",
# # #         "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
# # #         "configs/body_2d_keypoint/rtmpose/coco/"
# # #         "rtmpose-s_8xb256-420e_coco-256x192.py",
# # #     ),
# # #     "m": (
# # #         "rtmpose-m_8xb256-420e_coco-256x192.py",
# # #         "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
# # #         "configs/body_2d_keypoint/rtmpose/coco/"
# # #         "rtmpose-m_8xb256-420e_coco-256x192.py",
# # #     ),
# # #     "l": (
# # #         "rtmpose-l_8xb256-420e_coco-384x288.py",
# # #         "https://raw.githubusercontent.com/open-mmlab/mmpose/main/"
# # #         "configs/body_2d_keypoint/rtmpose/coco/"
# # #         "rtmpose-l_8xb256-420e_coco-384x288.py",
# # #     ),
# # # }

# # # HRNET_MODELS = {
# # #     "w32_256": {
# # #         "config": "td-hm_hrnet-w32_8xb32-210e_coco-256x192.py",
# # #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# # #                       "hrnet_w32_coco_256x192-8e0b4c43_20200708.pth",
# # #     },
# # #     "w32_384": {
# # #         "config": "td-hm_hrnet-w32_8xb32-210e_coco-384x288.py",
# # #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# # #                       "hrnet_w32_coco_384x288-5ed6c1c9_20200708.pth",
# # #     },
# # #     "w48_256": {
# # #         "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
# # #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# # #                       "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
# # #     },
# # #     "w48_384": {
# # #         "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
# # #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# # #                       "hrnet_w48_coco_384x288-033f4f36_20200708.pth",
# # #     },
# # # }


# # # import urllib.request
# # # from pathlib import Path

# # # import zipfile
# # # import urllib.request
# # # from pathlib import Path
# # # import shutil
# # # import tempfile


# # # def ensure_mmpose_configs(config_root: Path = Path("models/mmpose_configs")) -> Path:
# # #     """
# # #     Ensure full MMPose configs tree is available locally.
# # #     Downloads and extracts configs/ from the official GitHub repo if missing.
# # #     """

# # #     config_root = Path(config_root)
# # #     configs_dir = config_root / "configs"

# # #     if configs_dir.exists():
# # #         return configs_dir

# # #     print("[RTMPose] Downloading full MMPose configs...")

# # #     url = "https://github.com/open-mmlab/mmpose/archive/refs/heads/main.zip"

# # #     with tempfile.TemporaryDirectory() as tmpdir:
# # #         zip_path = Path(tmpdir) / "mmpose.zip"
# # #         urllib.request.urlretrieve(url, zip_path)

# # #         with zipfile.ZipFile(zip_path, "r") as zf:
# # #             for member in zf.namelist():
# # #                 if "mmpose-main/configs/" in member:
# # #                     zf.extract(member, tmpdir)

# # #         extracted_root = Path(tmpdir) / "mmpose-main" / "configs"
# # #         shutil.copytree(extracted_root, configs_dir)

# # #     print("[RTMPose] Configs ready:", configs_dir)
# # #     return configs_dir


# # # def ensure_rtmpose_config(
# # #     model_size: str,
# # #     config_dir: str | Path = "models/mmpose_configs/rtmpose",
# # # ) -> Path:
# # #     """
# # #     Ensure RTMPose config file is present locally.
# # #     Downloads it from the official mmpose GitHub repo if missing.
# # #     """

# # #     if model_size not in RTMPOSE_CONFIGS:
# # #         raise ValueError(
# # #             f"Invalid model_size '{model_size}'. "
# # #             f"Choose from {list(RTMPOSE_CONFIGS.keys())}"
# # #         )

# # #     filename, url = RTMPOSE_CONFIGS[model_size]

# # #     config_dir = Path(config_dir)
# # #     config_dir.mkdir(parents=True, exist_ok=True)

# # #     config_path = config_dir / filename

# # #     if not config_path.exists():
# # #         print(f"[RTMPose] Downloading config: {filename}")
# # #         print(f"[RTMPose] Source: {url}")
# # #         urllib.request.urlretrieve(url, config_path)

# # #     return config_path


# # # class RTMPoseEstimator:
# # #     def __init__(self, model_variant='w48_256'):
# # #         """
# # #         model_variant: 'w32_256', 'w32_384', 'w48_256', 'w48_384'
# # #         """

# # #         # ------------------------------------------------------------
# # #         # CONFIG / CHECKPOINT MAPS
# # #         # ------------------------------------------------------------
# # #         # config_map = {
# # #         #     't': 'rtmpose-t_8xb256-420e_coco-256x192.py',
# # #         #     's': 'rtmpose-s_8xb256-420e_coco-256x192.py',
# # #         #     'm': 'rtmpose-m_8xb256-420e_coco-256x192.py',
# # #         #     'l': 'rtmpose-l_8xb256-420e_coco-384x288.py'
# # #         # }

# # #         # checkpoint_map = {
# # #         #     't': 'rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # #         #     's': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # #         #     'm': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth',
# # #         #     'l': 'rtmpose-l_simcc-coco_pt-aic-coco_420e-384x288.pth'
# # #         # }

# # #         # if model_size not in config_map:
# # #         #     raise ValueError(f"Invalid model_size '{model_size}'. Choose from {list(config_map.keys())}")

# # #         # # ------------------------------------------------------------
# # #         # # RESOLVE CONFIG PATH FROM INSTALLED MMPOSE PACKAGE
# # #         # # ------------------------------------------------------------
# # #         # configs_root = ensure_mmpose_configs(Path("models/mmpose_configs"))

# # #         # config_file = (
# # #         #     configs_root
# # #         #     / "body_2d_keypoint"
# # #         #     / "topdown_heatmap" #"rtmpose"
# # #         #     / "coco"
# # #         #     / RTMPOSE_CONFIGS[model_size][0]
# # #         # )

# # #         # if not config_file.exists():
# # #         #     raise FileNotFoundError(f"RTMPose config not found: {config_file}")

# # #         # # ------------------------------------------------------------
# # #         # # CHECKPOINT (AUTO-DOWNLOADED & CACHED)
# # #         # # ------------------------------------------------------------
# # #         # checkpoint_file = (
# # #         #     # "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
# # #         #     # "rtmpose-s_simcc-coco_420e-256x192.pth"
# # #         #     "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
# # #         # )

# # #         # checkpoint_file = r'C:\Users\User\Downloads\td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

# # #         variant = HRNET_MODELS[model_variant]

# # #         configs_root = ensure_mmpose_configs(Path("models/mmpose_configs"))

# # #         config_file = (
# # #             configs_root
# # #             / "body_2d_keypoint"
# # #             / "topdown_heatmap"
# # #             / "coco"
# # #             / variant["config"]
# # #         )

# # #         checkpoint_file = variant["checkpoint"]


# # #         # ------------------------------------------------------------
# # #         # INITIALIZE RTMPOSE MODEL
# # #         # ------------------------------------------------------------
# # #         self.model = init_model(
# # #             str(config_file),
# # #             checkpoint_file,
# # #             device="cuda:0"
# # #         )

# # #         self.model_name = f"RTMPose-{model_variant.upper()}"

# # #         # ------------------------------------------------------------
# # #         # PERSON DETECTOR (YOLOv8)
# # #         # ------------------------------------------------------------
# # #         self.detector = YOLO("yolov8n.pt")

# # #     # ============================================================
# # #     # INFERENCE
# # #     # ============================================================

# # #     def predict(self, image, conf_threshold=0.25):
# # #         """
# # #         Returns COCO-format keypoints for ALL detected persons
# # #         """

# # #         det_results = self.detector(
# # #             image,
# # #             classes=[0],          # person
# # #             conf=conf_threshold,
# # #             verbose=False
# # #         )

# # #         if len(det_results[0].boxes) == 0:
# # #             return []

# # #         bboxes = []
# # #         person_confs = []

# # #         for i in range(len(det_results[0].boxes)):
# # #             bbox = det_results[0].boxes.xyxy[i].cpu().numpy()
# # #             conf = det_results[0].boxes.conf[i].cpu().numpy()

# # #             # MMPose expects [x1, y1, x2, y2, score]
# # #             # bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 1.0])
# # #             bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
# # #             person_confs.append(conf)

# # #         bboxes_array = np.asarray(bboxes, dtype=np.float32)

# # #         pose_results = inference_topdown(
# # #             self.model,
# # #             image,
# # #             bboxes=bboxes_array
# # #         )

# # #         detections = []

# # #         for i, pose_result in enumerate(pose_results):
# # #             keypoints = pose_result.pred_instances.keypoints[0]           # [17, 2]
# # #             scores = pose_result.pred_instances.keypoint_scores[0]        # [17]

# # #             keypoints_with_scores = np.concatenate(
# # #                 [keypoints, scores[:, None]],
# # #                 axis=1
# # #             )  # [17, 3]

# # #             detections.append({
# # #                 "keypoints": keypoints_with_scores,
# # #                 "score": float(person_confs[i]),
# # #                 "bbox": det_results[0].boxes.xyxy[i].cpu().numpy()
# # #             })

# # #         return detections

# # #     # ============================================================
# # #     # CONVENIENCE WRAPPERS
# # #     # ============================================================

# # #     def predict_single(self, image):
# # #         """Return highest-confidence person only"""
# # #         detections = self.predict(image)
# # #         if not detections:
# # #             return None
# # #         return max(detections, key=lambda x: x["score"])

# # #     def predict_batch(self, images):
# # #         """Process multiple images sequentially"""
# # #         return [self.predict(img) for img in images]

# # #     # ============================================================
# # #     # BENCHMARKING
# # #     # ============================================================

# # #     def benchmark(self, image, num_runs=100):
# # #         """Measure average inference latency"""

# # #         for _ in range(10):  # warmup
# # #             self.predict(image)

# # #         start = time.time()
# # #         for _ in range(num_runs):
# # #             self.predict(image)
# # #         end = time.time()

# # #         avg_time = (end - start) / num_runs

# # #         return {
# # #             "model": self.model_name,
# # #             "avg_time_ms": avg_time * 1000,
# # #             "fps": 1.0 / avg_time
# # #         }


# # # # ============================================================
# # # # EXAMPLE USAGE
# # # # ============================================================

# # # if __name__ == "__main__":

# # #     estimator = RTMPoseEstimator(model_variant='w48_256')

# # #     image = cv2.imread(
# # #         r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\Coco\val2017\000000000785.jpg"
# # #     )

# # #     all_people = estimator.predict(image)

# # #     print(f"Detected {len(all_people)} people")
# # #     for i, person in enumerate(all_people):
# # #         print(f"  Person {i+1}: confidence {person['score']:.3f}")

# # #     best_person = estimator.predict_single(image)
# # #     if best_person:
# # #         print(f"\nBest person confidence: {best_person['score']:.3f}")

# # #     # ------------------------------------------------------------
# # #     # VISUALIZATION
# # #     # ------------------------------------------------------------
# # #     annotated = image.copy()
# # #     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# # #     for idx, person in enumerate(all_people):
# # #         color = colors[idx % len(colors)]

# # #         for x, y, conf in person["keypoints"]:
# # #             if conf > 0.5:
# # #                 cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

# # #         x1, y1, x2, y2 = person["bbox"]
# # #         cv2.rectangle(
# # #             annotated,
# # #             (int(x1), int(y1)),
# # #             (int(x2), int(y2)),
# # #             color,
# # #             2
# # #         )

# # #     cv2.imshow("RTMPose - All People", annotated)
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()

# # #     # ------------------------------------------------------------
# # #     # BENCHMARK
# # #     # ------------------------------------------------------------
# # #     results = estimator.benchmark(image)
# # #     print("\nBenchmark Results:")
# # #     print(f"Model: {results['model']}")
# # #     print(f"Average Time: {results['avg_time_ms']:.2f} ms")
# # #     print(f"FPS: {results['fps']:.2f}")


# # import cv2
# # import numpy as np
# # import time
# # from pathlib import Path
# # import urllib.request
# # import zipfile
# # import tempfile
# # import shutil

# # from ultralytics import YOLO
# # from mmpose.apis import init_model, inference_topdown


# # # ============================================================
# # # HRNet MODEL REGISTRY
# # # ============================================================

# # HRNET_MODELS = {
# #     "w32_256": {
# #         "config": "td-hm_hrnet-w32_8xb32-210e_coco-256x192.py",
# #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# #                       "hrnet_w32_coco_256x192-8e0b4c43_20200708.pth",
# #     },
# #     "w32_384": {
# #         "config": "td-hm_hrnet-w32_8xb32-210e_coco-384x288.py",
# #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# #                       "hrnet_w32_coco_384x288-5ed6c1c9_20200708.pth",
# #     },
# #     "w48_256": {
# #         "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
# #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# #                       "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
# #     },
# #     "w48_384": {
# #         "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
# #         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
# #                       "hrnet_w48_coco_384x288-033f4f36_20200708.pth",
# #     },
# # }


# # # ============================================================
# # # ENSURE MMPOSE CONFIGS ARE AVAILABLE LOCALLY
# # # ============================================================

# # def ensure_mmpose_configs(root: Path = Path("models/mmpose_configs")) -> Path:
# #     configs_dir = root / "configs"
# #     if configs_dir.exists():
# #         return configs_dir

# #     print("[Pose] Downloading MMPose configs...")
# #     url = "https://github.com/open-mmlab/mmpose/archive/refs/heads/main.zip"

# #     with tempfile.TemporaryDirectory() as tmp:
# #         zip_path = Path(tmp) / "mmpose.zip"
# #         urllib.request.urlretrieve(url, zip_path)

# #         with zipfile.ZipFile(zip_path, "r") as zf:
# #             for name in zf.namelist():
# #                 if "mmpose-main/configs/" in name:
# #                     zf.extract(name, tmp)

# #         extracted = Path(tmp) / "mmpose-main" / "configs"
# #         shutil.copytree(extracted, configs_dir)

# #     print("[Pose] Configs ready:", configs_dir)
# #     return configs_dir


# # # ============================================================
# # # POSE ESTIMATOR
# # # ============================================================

# # class PoseEstimator:
# #     def __init__(
# #         self,
# #         model_variant: str = "w48_256",
# #         use_yolo: bool = True,
# #         device: str = "cuda:0",
# #     ):
# #         """
# #         model_variant: one of HRNET_MODELS keys
# #         use_yolo: if True, run YOLO person detection first
# #         """

# #         if model_variant not in HRNET_MODELS:
# #             raise ValueError(f"Invalid model_variant: {model_variant}")

# #         self.use_yolo = use_yolo

# #         # Resolve config
# #         configs_root = ensure_mmpose_configs()
# #         variant = HRNET_MODELS[model_variant]

# #         config_file = (
# #             configs_root
# #             / "body_2d_keypoint"
# #             / "topdown_heatmap"
# #             / "coco"
# #             / variant["config"]
# #         )

# #         # Init pose model
# #         self.model = init_model(
# #             str(config_file),
# #             variant["checkpoint"],
# #             device=device,
# #         )

# #         self.model_name = f"HRNet-{model_variant.upper()}"

# #         # Optional YOLO detector
# #         self.detector = YOLO("yolov8n.pt") if use_yolo else None


# #     # ========================================================
# #     # INFERENCE
# #     # ========================================================

# #     def predict(self, image, conf_threshold=0.25):
# #         """
# #         Returns a list of detections:
# #         {
# #             keypoints: [17, 3],
# #             score: float,
# #             bbox: [x1, y1, x2, y2] or full image
# #         }
# #         """

# #         # ----------------------------------------------------
# #         # CASE 1: YOLO OFF → pose on full image
# #         # ----------------------------------------------------
# #         if not self.use_yolo:
# #             pose_results = inference_topdown(self.model, image)

# #             detections = []
# #             for pose in pose_results:
# #                 kpts = pose.pred_instances.keypoints[0]
# #                 scores = pose.pred_instances.keypoint_scores[0]

# #                 detections.append({
# #                     "keypoints": np.concatenate([kpts, scores[:, None]], axis=1),
# #                     "score": float(scores.mean()),
# #                     "bbox": np.array([0, 0, image.shape[1], image.shape[0]])
# #                 })
# #             return detections

# #         # ----------------------------------------------------
# #         # CASE 2: YOLO ON → pose per bounding box
# #         # ----------------------------------------------------
# #         det = self.detector(
# #             image,
# #             classes=[0],
# #             conf=conf_threshold,
# #             verbose=False
# #         )

# #         if len(det[0].boxes) == 0:
# #             return []

# #         bboxes = det[0].boxes.xyxy.cpu().numpy()

# #         pose_results = inference_topdown(
# #             self.model,
# #             image,
# #             bboxes=bboxes
# #         )

# #         detections = []
# #         for i, pose in enumerate(pose_results):
# #             kpts = pose.pred_instances.keypoints[0]
# #             scores = pose.pred_instances.keypoint_scores[0]

# #             detections.append({
# #                 "keypoints": np.concatenate([kpts, scores[:, None]], axis=1),
# #                 "score": float(det[0].boxes.conf[i]),
# #                 "bbox": bboxes[i]
# #             })

# #         return detections


# #     # ========================================================
# #     # HELPERS
# #     # ========================================================

# #     def predict_single(self, image):
# #         detections = self.predict(image)
# #         return max(detections, key=lambda x: x["score"]) if detections else None

# #     def benchmark(self, image, runs=50):
# #         for _ in range(10):
# #             self.predict(image)

# #         t0 = time.time()
# #         for _ in range(runs):
# #             self.predict(image)
# #         dt = (time.time() - t0) / runs

# #         return {
# #             "model": self.model_name,
# #             "avg_time_ms": dt * 1000,
# #             "fps": 1.0 / dt,
# #         }


# # # ============================================================
# # # EXAMPLE USAGE
# # # ============================================================

# # if __name__ == "__main__":

# #     estimator = PoseEstimator(
# #         model_variant="w48_256",
# #         use_yolo=True,   # toggle here
# #     )

# #     image = cv2.imread(
# #         r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\Coco\val2017\000000001000.jpg" #000000000785.jpg"
# #     )

# #     detections = estimator.predict(image)
# #     print(f"Detected {len(detections)} people")

# #     vis = image.copy()
# #     for person in detections:
# #         for x, y, c in person["keypoints"]:
# #             if c > 0.5:
# #                 cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

# #         x1, y1, x2, y2 = person["bbox"]
# #         cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# #     cv2.imshow("Pose Estimation", vis)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time
# from pathlib import Path
# import urllib.request
# import zipfile
# import tempfile
# import shutil

# from ultralytics import YOLO
# from mmpose.apis import init_model, inference_topdown


# # ============================================================
# # HRNet MODEL REGISTRY (COCO TOP-DOWN)
# # ============================================================

# HRNET_MODELS = {
#     "w32_256": {
#         "config": "td-hm_hrnet-w32_8xb32-210e_coco-256x192.py",
#         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
#                       "hrnet_w32_coco_256x192-8e0b4c43_20200708.pth",
#     },
#     "w32_384": {
#         "config": "td-hm_hrnet-w32_8xb32-210e_coco-384x288.py",
#         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
#                       "hrnet_w32_coco_384x288-5ed6c1c9_20200708.pth",
#     },
#     "w48_256": {
#         "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
#         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
#                       "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
#     },
#     "w48_384": {
#         "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
#         "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/"
#                       "hrnet_w48_coco_384x288-033f4f36_20200708.pth",
#     },
# }


# # ============================================================
# # ENSURE MMPOSE CONFIGS ARE AVAILABLE LOCALLY
# # ============================================================

# def ensure_mmpose_configs(root: Path = Path("models/mmpose_configs")) -> Path:
#     configs_dir = root / "configs"
#     if configs_dir.exists():
#         return configs_dir

#     print("[MMPose] Downloading config repository...")
#     url = "https://github.com/open-mmlab/mmpose/archive/refs/heads/main.zip"

#     with tempfile.TemporaryDirectory() as tmp:
#         zip_path = Path(tmp) / "mmpose.zip"
#         urllib.request.urlretrieve(url, zip_path)

#         with zipfile.ZipFile(zip_path, "r") as zf:
#             for name in zf.namelist():
#                 if "mmpose-main/configs/" in name:
#                     zf.extract(name, tmp)

#         extracted = Path(tmp) / "mmpose-main" / "configs"
#         shutil.copytree(extracted, configs_dir)

#     print("[MMPose] Configs ready:", configs_dir)
#     return configs_dir


# # ============================================================
# # TOP-DOWN POSE ESTIMATOR (EVALUATOR-COMPATIBLE)
# # ============================================================

# class MMPoseTopDownEstimator:
#     """
#     Drop-in compatible with models_to_test:
#       predict(image, gt_boxes=None)
#     """

#     def __init__(
#         self,
#         model_variant: str = "w48_256",
#         use_yolo: bool = True,
#         device: str = "cuda:0",
#     ):
#         if model_variant not in HRNET_MODELS:
#             raise ValueError(f"Invalid model_variant: {model_variant}")

#         self.use_yolo = use_yolo

#         # Resolve config
#         configs_root = ensure_mmpose_configs()
#         variant = HRNET_MODELS[model_variant]

#         self.config_file = (
#             configs_root
#             / "body_2d_keypoint"
#             / "topdown_heatmap"
#             / "coco"
#             / variant["config"]
#         )

#         self.checkpoint = variant["checkpoint"]

#         self.model = init_model(
#             str(self.config_file),
#             self.checkpoint,
#             device=device,
#         )

#         self.model_name = f"HRNet-{model_variant.upper()}"

#         self.detector = YOLO("yolov8n.pt") if use_yolo else None


#     # ========================================================
#     # MAIN API (EVALUATOR EXPECTS THIS)
#     # ========================================================

#     def predict(self, image, gt_boxes=None, conf_threshold=0.25):
#         """
#         Args:
#             image: np.ndarray (H, W, 3)
#             gt_boxes: Optional[np.ndarray] of shape (N, 4)
#         Returns:
#             List[{
#                 keypoints: (17, 3),
#                 score: float,
#                 bbox: (4,)
#             }]
#         """

#         # ----------------------------------------------------
#         # ORACLE MODE (GT BOXES PROVIDED)
#         # ----------------------------------------------------
#         if gt_boxes is not None:
#             return self._predict_from_boxes(image, gt_boxes)

#         # ----------------------------------------------------
#         # NO DETECTOR → FULL IMAGE
#         # ----------------------------------------------------
#         if not self.use_yolo:
#             pose_results = inference_topdown(self.model, image)

#             detections = []
#             for pose in pose_results:
#                 kpts = pose.pred_instances.keypoints[0]
#                 scores = pose.pred_instances.keypoint_scores[0]

#                 detections.append({
#                     "keypoints": np.concatenate([kpts, scores[:, None]], axis=1),
#                     "score": float(scores.mean()),
#                     "bbox": np.array([0, 0, image.shape[1], image.shape[0]])
#                 })

#             return detections

#         # ----------------------------------------------------
#         # STANDARD MODE → YOLO PERSON DETECTOR
#         # ----------------------------------------------------
#         det = self.detector(
#             image,
#             classes=[0],
#             conf=conf_threshold,
#             verbose=False
#         )

#         if len(det[0].boxes) == 0:
#             return []

#         bboxes = det[0].boxes.xyxy.cpu().numpy()
#         scores_det = det[0].boxes.conf.cpu().numpy()

#         return self._predict_from_boxes(image, bboxes, scores_det)


#     # ========================================================
#     # INTERNAL: BOX-BASED INFERENCE
#     # ========================================================

#     def _predict_from_boxes(self, image, boxes, det_scores=None):
#         pose_results = inference_topdown(
#             self.model,
#             image,
#             bboxes=boxes
#         )

#         detections = []
#         for i, pose in enumerate(pose_results):
#             kpts = pose.pred_instances.keypoints[0]
#             scores = pose.pred_instances.keypoint_scores[0]

#             detections.append({
#                 "keypoints": np.concatenate([kpts, scores[:, None]], axis=1),
#                 "score": float(det_scores[i]) if det_scores is not None else float(scores.mean()),
#                 "bbox": boxes[i]
#             })

#         return detections


#     # ========================================================
#     # OPTIONAL UTILITIES
#     # ========================================================

#     def predict_single(self, image, gt_boxes=None):
#         dets = self.predict(image, gt_boxes)
#         return max(dets, key=lambda x: x["score"]) if dets else None

#     def benchmark(self, image, runs=50):
#         for _ in range(10):
#             self.predict(image)

#         t0 = time.time()
#         for _ in range(runs):
#             self.predict(image)
#         dt = (time.time() - t0) / runs

#         return {
#             "model": self.model_name,
#             "avg_time_ms": dt * 1000,
#             "fps": 1.0 / dt,
#         }

# # ============================================================
# # EXAMPLE USAGE
# # ============================================================

# if __name__ == "__main__":

#     estimator = MMPoseTopDownEstimator(
#         model_variant="w48_256",
#         use_yolo=True,   # toggle here
#     )

#     image = cv2.imread(
#         r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\PoseEstimation\Coco\val2017\000000000785.jpg" #000000001000.jpg" #
#     )

#     detections = estimator.predict(image)
#     print(f"Detected {len(detections)} people")

#     vis = image.copy()
#     for person in detections:
#         for x, y, c in person["keypoints"]:
#             if c > 0.5:
#                 cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

#         x1, y1, x2, y2 = person["bbox"]
#         cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

#     cv2.imshow("Pose Estimation", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


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
        "config": "td-hm_hrnet-w32_8xb32-210e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w32_coco_256x192-8e0b4c43_20200708.pth"
        ),
    },
    "w32_384": {
        "config": "td-hm_hrnet-w32_8xb32-210e_coco-384x288.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w32_coco_384x288-5ed6c1c9_20200708.pth"
        ),
    },
    "w48_256": {
        "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        ),
    },
    "w48_384": {
        "config": "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w48_coco_384x288-033f4f36_20200708.pth"
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