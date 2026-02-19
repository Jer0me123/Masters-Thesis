import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import List, Dict, Any, Tuple, Set
import torch
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import argparse
import gc
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ----------------- Helper Functions -----------------

# --- Worker Function (CPU-intensive: Visualization and JSON Save) ---
def draw_and_save_image(
    image: Image.Image, 
    instances_info: List[Dict[str, Any]], 
    save_path_img: str, 
    save_path_json: str,
    color_map: Dict[str, Tuple[int, int, int]],
    white_background: bool = False
) -> Tuple[str, str]:
    """
    Core function to perform visualization (CV2/Numpy) and save detections to JSON.
    This function is run inside the ProcessPoolExecutor.
    """
    # Convert PIL Image to BGR for OpenCV drawing
    if white_background:
        width, height = image.size
        # np.full creates a new array filled with the specified value (255 for white)
        # BGR format is (Height, Width, Channels)
        img_np = np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        # Original logic
        img_np = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    
    # Define new text parameters
    FONT_SCALE = 0.4    # Reduced font size for smaller labels
    FONT_THICKNESS = 1
    BOX_THICKNESS = 2
    TEXT_PADDING = 2
    
    # ----------------- Data for JSON -----------------
    json_data = {}
    
    for instance in instances_info:
        # Box format: [xmin, ymin, xmax, ymax]
        box = instance["box"]
        label = instance["label"]
        score = instance["score"]

        # Get color based on label or use a default if not found
        formatted_label = label.title()
        color = color_map.get(formatted_label, (0, 255, 0))  # Default to green (BGR)

        # Draw box
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(img_np, pt1, pt2, color, BOX_THICKNESS)

        # --- Label Placement Logic ---
        text = f"{label}: {score:.2f}"
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        # Determine the starting coordinates for the background rectangle
        x_start = pt1[0]
        y_start = pt1[1]

        # Calculate the top-left (pt_bg1) and bottom-right (pt_bg2) corners of the background
        pt_bg1 = (x_start, y_start) 
        pt_bg2 = (x_start + w + 2 * TEXT_PADDING, y_start + h + baseline + 2 * TEXT_PADDING)

        # Draw background rectangle for text.
        cv2.rectangle(
            img_np, 
            pt_bg1, 
            pt_bg2, 
            color, 
            -1 # Filled rectangle
        )
        
        # Determine the text position
        text_x = x_start + TEXT_PADDING
        text_y = y_start + h + TEXT_PADDING # y_start (top line) + h (text height) + padding

        # Draw text (Black text)
        cv2.putText(
            img_np, 
            text, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            FONT_SCALE, 
            (0, 0, 0), # Black text color (BGR)
            FONT_THICKNESS
        )
        
        # Aggregate detection data for JSON
        if label not in json_data:
             json_data[label] = []
        json_data[label].append({
            'box_xyxy': box, 
            'confidence': score
        })

    # Save annotated image
    cv2.imwrite(save_path_img, img_np)
    
    # Save detection data to a dedicated JSON file for this image
    with open(save_path_json, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    return save_path_img, save_path_json

# --- Intermediate Worker Function (Handles opening the file for the executor) ---
def process_and_save_single_image(
    result: Results,
    output_dir: Path,
    json_output_dir: Path,
    class_names: Dict[int, str],
    color_map: Dict[str, Tuple[int, int, int]],
    conf_thresh: float,
    executor: ProcessPoolExecutor,
    white_background: bool = False,
    excluded_classes_set: Set[str] = None,
    included_classes_set: Set[str] = None
) -> Tuple[Future, int] or None:
    """
    Prepares data from a single YOLO result and submits the visualization/save task 
    to the ProcessPoolExecutor. Returns (Future, detection_count) or None on failure.
    """
    image_path = Path(result.path)
    base_name = image_path.stem
    
    # Load the original image file for visualization
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path} for visualization: {e}")
        return None 
    
    # Handle defaults
    if excluded_classes_set is None:
        excluded_classes_set = set()
    if included_classes_set is None:
        included_classes_set = set()
        
    instances_info: List[Dict[str, Any]] = []
    total_detections = 0
    
    if result.boxes is not None and result.boxes.xyxy.shape[0] > 0:
        # Get results on CPU as numpy for easy iteration
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Extract, filter, and structure the data
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf >= conf_thresh: # Apply confidence threshold
                
                label = class_names[class_id]
                # Normalize the label (Title Case) for robust comparison
                formatted_label = label.replace('_', ' ').title() 

                # Class Inclusion Check (Highest Precedence)
                if included_classes_set and formatted_label not in included_classes_set:
                    continue # Skip if this label is not in the required list

                # Class Exclusion Check (Fallback)
                if formatted_label in excluded_classes_set:
                    continue # Skip this instance if it is explicitly excluded

                total_detections += 1
                instances_info.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'score': float(conf),
                    'label': label,
                })
    
    save_path_img = output_dir / f"{base_name}.png"
    save_path_json = json_output_dir / f"{base_name}.json"

    # Submit save task to the process pool
    future = executor.submit(
        draw_and_save_image, 
        image_pil, 
        instances_info, 
        str(save_path_img),
        str(save_path_json),
        color_map,
        white_background
    )
    return future, total_detections


# ---------------- Main Object Detection Function ----------------
def detect_objects(
    image_dir: str,
    output_dir: str,
    model_path: str, # YOLOv8 model file path (e.g., 'yolov8x-oiv7.pt')
    color_json: str,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    imgsz: int = 640,
    resume: bool = True,
    white_background: bool = False,
    exclude_classes: str = "",
    include_classes: str = ""
):
    # Setup device and directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow.")

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    json_output_dir = output_dir / "json_detections"
    os.makedirs(json_output_dir, exist_ok=True)
    
    # --- 1. Load Model and Colors ---
    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    model.eval()
    class_names = model.names
    
    with open(color_json, "r") as f:
        color_map = {k.title(): tuple(v) for k, v in json.load(f).items()} 

    # Process included classes
    included_classes_set = set()
    if include_classes:
        # Split by comma, strip spaces, and normalize to Title Case set for fast lookup
        included_classes_set = {c.strip().title() for c in include_classes.split(',')}
        print(f"Only including the following normalized classes: {included_classes_set}")

    # Process excluded classes (only if include is NOT used)
    excluded_classes_set = set()
    if not included_classes_set and exclude_classes:
        # Split by comma, strip spaces, and normalize to Title Case set for fast lookup
        excluded_classes_set = {c.strip().title() for c in exclude_classes.split(',')}
        print(f"Excluding the following normalized classes: {excluded_classes_set}")
    
    # --- 2. Filter Image Paths (Resume Logic) ---
    image_dir_path = Path(image_dir)
    all_image_paths = sorted(
        [str(p) for ext in (".jpg", ".jpeg", ".png") for p in image_dir_path.glob(f"*{ext}")]
    )
    
    # Identify processed images by checking for their JSON files
    if resume:
        processed_json_stems = {p.stem for p in json_output_dir.glob("*.json")}
        
        # Filter the list to only include paths that haven't been processed
        paths_to_process = [
            path for path in all_image_paths if Path(path).stem not in processed_json_stems
        ]
        
        processed_count = len(all_image_paths) - len(paths_to_process)
        print(f"Resuming: Found {processed_count} processed files. {len(paths_to_process)} images left to process.")
    else:
        paths_to_process = all_image_paths
    
    if not paths_to_process:
        print("All images already processed. Exiting.")
        return
    
    # --- 3. Chunk the Paths to Limit File Handles and GPU Memory ---
    CHUNK_SIZE = 32
    total_images_to_process = len(paths_to_process)
    
    # Split the list into chunks of CHUNK_SIZE
    chunks = [paths_to_process[i:i + CHUNK_SIZE] 
              for i in range(0, total_images_to_process, CHUNK_SIZE)]
    
    print(f"Starting batched YOLO inference on {total_images_to_process} images in {len(chunks)} chunks (Chunk Size: {CHUNK_SIZE}).")

    all_futures: List[Future] = []
    total_detections = 0
    
    # --- 4. Loop Through Chunks, Run YOLO, and Offload Saving ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
          tqdm(total=total_images_to_process, desc="YOLOv8 Inference (GPU) & Submission (CPU)") as pbar:

        for chunk_idx, chunk in enumerate(chunks):
            with torch.no_grad():
                results_generator = model.predict(
                    source=chunk,
                    conf=conf_thresh,
                    imgsz=imgsz,
                    device=device,
                    batch=batch_size,
                    stream=True,
                    save=False, 
                    verbose=False
                )

                for r in results_generator:
                    
                    # Offload visualization and saving to the CPU process pool
                    future_result = process_and_save_single_image(
                        r, output_dir, json_output_dir, class_names, color_map, conf_thresh, executor, white_background,
                        excluded_classes_set,
                        included_classes_set
                    )
                    
                    if future_result:
                        future, det_count = future_result
                        all_futures.append(future)
                        total_detections += det_count
                    
                    pbar.update(1) # Update the progress bar for each processed image

            del results_generator
            torch.cuda.empty_cache()
            gc.collect()

        # Wait for all visualization/save tasks to complete
        if all_futures:
            for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Saving final results (CPU)"):
                future.result()
                
    torch.cuda.empty_cache()
    print(f"\n✅ YOLOv8 detection completed successfully!")
    print(f"✅ Total detections found (above {conf_thresh}): {total_detections}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Object Detection using YOLOv8 (Open Images V7)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detected images and JSON.")
    parser.add_argument("--model_path", type=str, default="yolov8x-oiv7.pt", help="Path/ID to the YOLOv8 model weights.")
    parser.add_argument("--color_json", type=str, required=True, help="Path to the LVIS/Open Images color map JSON.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images per batch for GPU inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading (now unused).")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Bounding Box score threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference (YOLOv8 default is 640).")
    parser.add_argument("--white_background", action="store_true", help="If set, bounding boxes will be drawn on a pure white background instead of the original image.")
    parser.add_argument(
        "--exclude_classes", 
        type=str, 
        default="", 
        help="Comma-separated list of class names (e.g., 'Human face,Dog') to exclude from visualization and JSON output. Ignored if --include_classes is used."
    )
    parser.add_argument(
        "--include_classes", 
        type=str, 
        default="", 
        help="Comma-separated list of class names (e.g., 'Man,Woman') to ONLY include in visualization and JSON output. Takes precedence over --exclude_classes."
    )
    args = parser.parse_args()
    
    detect_objects(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        color_json=args.color_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thresh=args.conf_thresh if args.conf_thresh is not None else 0.25,
        imgsz=args.imgsz,
        white_background=args.white_background,
        exclude_classes=args.exclude_classes,
        include_classes=args.include_classes
    )

# ---------------- Example Usage ----------------
# Yolov8x model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640

# Yolov8x White Background model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --white_background

# Example command to exclude specific classes:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --exclude_classes "Man, Woman, Human face"

# Example command to include specific classes only:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --include_classes "Man,Woman"