import os
import gc
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List, Set
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Lock

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ============================================================
# THREAD-SAFE JSONL WRITER
# ============================================================
class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        os.makedirs(path.parent, exist_ok=True)

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

# ============================================================
# CPU WORKER: RENDER + SAVE FOR ONE OPERATION
# ============================================================
def render_and_save(
    base_img: np.ndarray,
    instances_info: List[Dict[str, Any]],
    save_img_path: str,
    color_map: Dict[str, Tuple[int, int, int]],
    resize: Tuple[int, int] | None,
):
    h0, w0 = base_img.shape[:2]

    if resize is not None:
        target_w, target_h = resize
        sx = target_w / w0
        sy = target_h / h0
        img = cv2.resize(base_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = base_img
        sx = sy = 1.0
        target_w, target_h = w0, h0

    ref = min(target_w, target_h)
    FONT_SCALE = max(0.4, ref / 600)
    FONT_THICKNESS = max(1, int(ref / 400))
    BOX_THICKNESS = max(1, int(ref / 300))
    TEXT_PADDING = max(2, int(ref / 150))

    for inst in instances_info:
        x1, y1, x2, y2 = inst["box"]
        label = inst["label"]
        score = inst["score"]

        rx1 = int(x1 * sx)
        ry1 = int(y1 * sy)
        rx2 = int(x2 * sx)
        ry2 = int(y2 * sy)

        color = color_map.get(label.title(), (0, 255, 0))

        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, BOX_THICKNESS)

        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )

        cv2.rectangle(
            img,
            (rx1, ry1),
            (rx1 + tw + 2 * TEXT_PADDING, ry1 + th + baseline + 2 * TEXT_PADDING),
            color,
            -1,
        )

        cv2.putText(
            img,
            text,
            (rx1 + TEXT_PADDING, ry1 + th + TEXT_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
        )

    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    cv2.imwrite(save_img_path, img)

# ============================================================
# CPU WORKER: PROCESS IMAGE (RETURNS JSONL RECORD)
# ============================================================
def process_image(
    image: Image.Image,
    instances_info: List[Dict[str, Any]],
    rel_image_path: str,
    output_root: Path,
    operations: Set[str],
    color_map: Dict[str, Tuple[int, int, int]],
    resize: Tuple[int, int] | None,
):
    normal_base = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    white_base = np.full_like(normal_base, 255)

    h0, w0 = normal_base.shape[:2]

    record = {
        "image": rel_image_path,
        "image_size": [w0, h0],
        "detections": {},
    }

    if resize is not None:
        record["resized_image_size"] = list(resize)

    for inst in instances_info:
        entry = {
            "box_xyxy": inst["box"],
            "confidence": inst["score"],
        }

        if resize is not None:
            x1, y1, x2, y2 = inst["box"]
            rx1 = int(x1 * resize[0] / w0)
            ry1 = int(y1 * resize[1] / h0)
            rx2 = int(x2 * resize[0] / w0)
            ry2 = int(y2 * resize[1] / h0)
            entry["box_xyxy_resized"] = [rx1, ry1, rx2, ry2]

        record["detections"].setdefault(inst["label"], []).append(entry)

    for op in operations:
        base = normal_base if op == "normal" else white_base
        save_img = output_root / op / Path(rel_image_path).with_suffix(".png")
        render_and_save(base, instances_info, str(save_img), color_map, resize)

    return record

# ============================================================
# RESULT HANDLER
# ============================================================
def submit_result(
    r: Results,
    image_root: Path,
    output_root: Path,
    class_names: Dict[int, str],
    color_map: Dict[str, Tuple[int, int, int]],
    conf_thresh: float,
    executor: ThreadPoolExecutor,
    excluded_dirs: Set[str],
    excluded_classes: Set[str],
    included_classes: Set[str],
    operations: Set[str],
    resize: Tuple[int, int] | None,
    label_remap: Dict[str, str],
):
    img_path = Path(r.path)
    rel = img_path.relative_to(image_root)
    rel_dirs = {p.lower() for p in rel.parts[:-1]}
    if excluded_dirs & rel_dirs:
        return None

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    instances = []

    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for b, s, c in zip(boxes, scores, classes):
            if s < conf_thresh:
                continue

            raw_label = class_names[c]
            norm_label = raw_label.replace("_", " ").title()
            final_label = label_remap.get(norm_label, norm_label)

            if included_classes and final_label not in included_classes:
                continue
            if excluded_classes and final_label in excluded_classes:
                continue

            instances.append({
                "box": b.tolist(),
                "label": final_label,
                "score": float(s),
            })

    rel_str = str(rel).replace("\\", "/")

    return executor.submit(
        process_image,
        image,
        instances,
        rel_str,
        output_root,
        operations,
        color_map,
        resize,
    )

# ============================================================
# MAIN PIPELINE
# ============================================================
def detect_objects(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model_path).to(device).eval()
    class_names = model.names

    with open(args.color_json) as f:
        color_map = {k.title(): tuple(v) for k, v in json.load(f).items()}

    label_remap = {}
    if args.label_remap:
        with open(args.label_remap, "r", encoding="utf-8") as f:
            label_remap = {k.title(): v.title() for k, v in json.load(f).items()}

    image_root = Path(args.image_dir)
    output_root = Path(args.output_dir)
    jsonl_path = output_root / "detections.jsonl"
    jsonl_writer = JsonlWriter(jsonl_path)

    operations = set(args.operation)

    excluded_dirs = {d.lower() for d in args.exclude_dirs}
    excluded_classes = {c.title() for c in args.exclude_classes}

    if args.exclude_classes_file:
        with open(args.exclude_classes_file, "r", encoding="utf-8") as f:
            file_classes = {
                line.strip().title()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            }
        excluded_classes |= file_classes

    included_classes = {c.title() for c in args.include_classes}

    if args.include_classes_file:
        with open(args.include_classes_file, "r", encoding="utf-8") as f:
            file_classes = {
                line.strip().title()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            }
        included_classes |= file_classes

    completed = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                completed.add(json.loads(line)["image"])
    print(f"Found {len(completed)} completed images, skipping these.")

    all_images = [
        p for p in image_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and str(p.relative_to(image_root)).replace("\\", "/") not in completed
        and not ({x.lower() for x in p.relative_to(image_root).parts[:-1]} & excluded_dirs)
    ]

    chunks = [all_images[i:i + args.chunk_size]
              for i in range(0, len(all_images), args.chunk_size)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
         tqdm(total=len(all_images), desc="YOLO inference") as pbar:

        for chunk in chunks:
            futures: List[Future] = []

            with torch.no_grad():
                for r in model.predict(
                    source=[str(p) for p in chunk],
                    imgsz=args.imgsz,
                    conf=args.conf_thresh,
                    iou=args.iou_thresh,
                    augment=args.augment,
                    agnostic_nms=args.agnostic_nms,
                    batch=args.batch_size,
                    stream=True,
                    device=device,
                    verbose=False,
                ):
                    fut = submit_result(
                        r,
                        image_root,
                        output_root,
                        class_names,
                        color_map,
                        args.conf_thresh,
                        executor,
                        excluded_dirs,
                        excluded_classes,
                        included_classes,
                        operations,
                        args.resize,
                        label_remap,
                    )
                    if fut:
                        futures.append(fut)
                    pbar.update(1)

            for f in as_completed(futures):
                record = f.result()
                if record is not None:
                    jsonl_writer.write(record)

            torch.cuda.empty_cache()
            gc.collect()

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="yolov8x-oiv7.pt")
    parser.add_argument("--color_json", required=True)
    parser.add_argument("--label_remap", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--agnostic_nms", action="store_true")
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"])
    parser.add_argument("--exclude_classes", nargs="+", default=[])
    parser.add_argument("--exclude_classes_file", type=str, default=None, help="Path to a text file with one class name per line to exclude")
    parser.add_argument("--include_classes", nargs="+", default=[])
    parser.add_argument("--include_classes_file", type=str, default=None, help="Path to a text file with one class name per line to include")
    parser.add_argument("--operation", nargs="+", choices=["normal", "white_background"], default=["normal"])
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    detect_objects(parser.parse_args())

# ===========================================================
# EXAMPLE USAGE
# python ObjectDetection.py --image_dir path/to/images --output_dir path/to/output --model_path yolov8x-oiv7.pt --color_json path/to/color_map.json --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --operation normal white_background --resize 224 224 --label_remap path/to/label_remap.json

# --model_path yolov8x-oiv7.pt -> This is the YOLOv8x model trained on OpenImagesv7 dataset which has 601 classes, it was selected as it provided the best performance in terms of accuracy and number of classes covered when tested against the other models.
# --color_json path/to/color_map.json -> This is the color map json file which contains the mapping of class names to colors for rendering the bounding boxes.
# --label_remap path/to/label_remap.json -> This is an optional json file which contains the mapping of certain class names to other class names for remapping purposes. E.g. Remapping Man & Woman to Person.
# --imgsz 640 -> This is the image size used for inference, it was selected as a good balance between speed and accuracy. (Best parameter based on Hyperparameter tuning)
# --conf_thresh 0.1 -> This is the confidence threshold for filtering detections, (Best parameter based on Hyperparameter tuning)
# --iou_thresh 0.5 -> This is the IoU threshold for non-maximum suppression. (Best parameter based on Hyperparameter tuning)
# --augment -> This flag enables test time augmentation which can improve accuracy at the cost of speed. (Absent as best parameter based on Hyperparameter tuning)
# --agnostic_nms -> This flag enables class-agnostic NMS which can help in reducing false positives across classes. (Absent as best parameter based on Hyperparameter tuning)
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --exclude_classes [] -> This can be used to exclude specific classes from detection if needed.
# --include_classes [] -> This can be used to include only specific classes for detection if needed.
# --operation normal white_background -> This specifies the types of output images to generate, normal for original background and white_background for white background.
# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.

# python ObjectDetection.py --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\test" --model_path "yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh

# python ObjectDetection.py --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\test3" --model_path "yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --exclude_classes "Human face" --label_remap "label_remap.json"

# NOTE: The label_remap.json file is not complete and needs to be extended to include all the necessary class mappings.

# python ObjectDetection.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ObjectDetection" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs face_crops --resize 224 224 --operation normal white_background --label_remap "label_remap.json" --exclude_classes_file "exclude_classes.txt"

# python ObjectDetection.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ObjectDetection_LabelRestricted" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs face_crops --resize 224 224 --operation normal white_background --label_remap "label_remap_restricted.json" --exclude_classes_file "exclude_classes.txt"

# python ObjectDetection.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\ObjectDetection" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --label_remap "label_remap.json" --exclude_classes_file "exclude_classes.txt"

# python ObjectDetection.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\ObjectDetection_LabelRestricted" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --label_remap "label_remap_restricted.json" --exclude_classes_file "exclude_classes.txt"


# python ObjectDetection.py --image_dir "F:\ImageRetrieval\Coco" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Coco\ObjectDetection" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --label_remap "label_remap.json" --exclude_classes_file "exclude_classes.txt"

# python ObjectDetection.py --image_dir "F:\ImageRetrieval\Coco" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Coco\ObjectDetection_LabelRestricted" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --label_remap "label_remap_restricted.json" --exclude_classes_file "exclude_classes.txt"

