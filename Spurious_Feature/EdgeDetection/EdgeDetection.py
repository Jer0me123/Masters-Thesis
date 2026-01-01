import os
import glob
import json
import argparse
from pathlib import Path
from threading import Lock, Thread
from queue import Queue
import numpy as np
import cv2
from tqdm import tqdm
import urllib.request

SAM_MODELS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}

def ensure_sam_model(model_size: str, model_dir="sam_models") -> Path:
    model_name, url = SAM_MODELS[model_size]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    if not model_path.exists():
        print(f"[SAM] Downloading {model_name} ...")
        urllib.request.urlretrieve(url, model_path)

    return model_path

# ============================================================
# FAST RGB DECODER (TurboJPEG)
# ============================================================
class FastRGBDecoder:
    """
    High-performance RGB image decoder.

    This class mirrors the decoding strategy used in PixelPatchShufflingMeanRGB.py:
    - Prefer libjpeg-turbo for JPEGs (fastest path)
    - Fallback to OpenCV if TurboJPEG is unavailable
    - Always return RGB NumPy arrays (not BGR)

    This guarantees:
    - Consistent color ordering
    - Minimal CPU overhead
    - Deterministic preprocessing for reproducibility
    """
        
    def __init__(self):
        try:
            from turbojpeg import TurboJPEG
            self.jpeg = TurboJPEG(
                r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll"
            )
            self.use_turbo = True
            print("[Decoder] TurboJPEG enabled")
        except Exception:
            self.use_turbo = False
            print("[Decoder] TurboJPEG unavailable, using OpenCV")

    def load(self, path: str) -> np.ndarray | None:
        try:
            ext = os.path.splitext(path)[1].lower()
            if self.use_turbo and ext in {".jpg", ".jpeg"}:
                with open(path, "rb") as f:
                    return self.jpeg.decode(f.read())
            arr = cv2.imread(path)
            if arr is None:
                return None
            return arr[..., ::-1]
        except Exception:
            return None


# ============================================================
# ASYNC IMAGE WRITER
# ============================================================
class AsyncImageWriter:
    """
    Asynchronous image writer.

    This writer:
    - Uses a bounded queue
    - Writes images in background threads
    - Prevents compute stalls during PNG encoding

    This mirrors the exact strategy used in PixelPatchShufflingMeanRGB.py.
    """
       
    def __init__(self, num_workers=4, max_queue=8192):
        self.queue = Queue(maxsize=max_queue)
        self.stop_token = object()
        self.workers = []

        for _ in range(num_workers):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is self.stop_token:
                break
            path, arr = item
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, arr)
            self.queue.task_done()

    def submit(self, path: str, arr: np.ndarray):
        self.queue.put((path, arr))

    def close(self):
        self.queue.join()
        for _ in self.workers:
            self.queue.put(self.stop_token)


# ============================================================
# TRANSFORM MANIFEST (PER-TRANSFORM)
# ============================================================
class TransformManifest:
    """
    Persistent per-transform JSONL manifest.
    """

    def __init__(self, path: str, flush_every: int = 1): #512):
        self.path = path
        self.flush_every = flush_every
        self.lock = Lock()
        self.data = set()
        self.buffer = []

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.data.add(r["image"])

    def record(self, image: str):
        with self.lock:
            self.data.add(image)
            self.buffer.append({"image": image})
            if len(self.buffer) >= self.flush_every:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for r in self.buffer:
                f.write(json.dumps(r) + "\n")
        self.buffer.clear()


# ============================================================
# DATASET
# ============================================================
class ImageDataset:
    """
    Structure-preserving dataset enumerator.

    Responsibilities:
    -----------------
    - Recursive traversal of image_dir
    - Preserve relative directory structure
    - Exclude unwanted subdirectories (e.g. facemesh)
    - Skip images already recorded in manifest
    - Decode images exactly once per batch
    - Optionally resize images before edge detection
    """

    def __init__(self, image_dir, completed, exclude_dirs, resize):
        self.decoder = FastRGBDecoder()
        self.image_dir = image_dir
        self.resize = resize
        exclude_dirs = {d.lower() for d in exclude_dirs}

        self.paths = []
        for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
                continue

            rel = os.path.relpath(p, image_dir).replace("\\", "/")
            if rel in completed:
                continue

            self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def get_batch(self, i, bs):
        paths = self.paths[i:i + bs]
        arrays = []
        for p in paths:
            arr = self.decoder.load(p)
            if arr is None or arr.shape[0] <= 1 or arr.shape[1] <= 1:
                arr = np.zeros((224, 224, 3), dtype=np.uint8)
            elif self.resize is not None:
                arr = cv2.resize(arr, self.resize, interpolation=cv2.INTER_LINEAR)
            arrays.append(arr)
        return arrays, paths


# ============================================================
# EDGE METHODS
# ============================================================
def canny_edges(arr_rgb, thr1, thr2):
    """
    Classical Canny edge detector.

    Parameters:
    -----------
    thr1 : lower hysteresis threshold
    thr2 : upper hysteresis threshold

    Defaults match the exact values used in:
    https://github.com/boyazeng/understand_bias/blob/main/transformations/canny/transform.py

    This implementation intentionally uses OpenCV CPU Canny
    to match the paper's preprocessing exactly.
    """
        
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise and improve edge detection (In line with paper methodology - https://github.com/boyazeng/understand_bias/blob/main/transformations/canny/transform.py)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, thr1, thr2)


class SAMEdgeDetector:
    """
    SAM-based edge detector.

    Instead of gradient-based edges, this method:
    - Uses SAM to generate object masks
    - Converts mask boundaries into edge maps

    This aligns with the SAM-based edge transformation
    used in "Understanding Bias in Large-Scale Visual Datasets".
    """

    def __init__(self, model_type, device="cuda", points_per_side=32):
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        self.torch = torch
        model_path = ensure_sam_model(model_type)
        self.sam = sam_model_registry[model_type](checkpoint=model_path).to(device)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam, points_per_side=points_per_side
        )

    def detect_edge(self, arr_rgb):
        h, w = arr_rgb.shape[:2]
        with self.torch.no_grad():
            masks = self.mask_generator.generate(arr_rgb)

        label_map = np.zeros((h, w), dtype=np.int32)
        for i, ann in enumerate(sorted(masks, key=lambda x: x.get("predicted_iou", 0), reverse=True)):
            label_map[ann["segmentation"]] = i + 1

        edges = np.zeros((h, w), dtype=np.uint8)
        edges[1:-1, 1:-1] = (
            (label_map[1:-1, 1:-1] != label_map[:-2, 1:-1]) |
            (label_map[1:-1, 1:-1] != label_map[2:, 1:-1]) |
            (label_map[1:-1, 1:-1] != label_map[1:-1, :-2]) |
            (label_map[1:-1, 1:-1] != label_map[1:-1, 2:])
        ) * 255

        return edges


# ============================================================
# MAIN
# ============================================================
def main(args):
    """
    Main execution loop.

    High-level pipeline:
    --------------------
    1. Select edge method (Canny or SAM)
    2. Load per-transform manifest
    3. Build dataset of incomplete images
    4. Decode images in batches
    5. Compute edge maps
    6. Write results asynchronously
    7. Record completion immediately
    """

    if args.edge_method == "canny":
        out_subdir = "edges_canny"
    else:
        out_subdir = f"edges_sam_{args.sam_model_type}_pps{args.sam_points_per_side}"

    manifest_path = os.path.join(
        args.output_dir, out_subdir, "transform_manifest.jsonl"
    )
    manifest = TransformManifest(manifest_path)

    completed = manifest.data
    print(f"Skipping {len(completed)} completed images.")

    resize = tuple(args.resize) if args.resize else None

    dataset = ImageDataset(
        args.image_dir,
        completed,
        args.exclude_dirs,
        resize,
    )
    print(f"Images to process: {len(dataset)}")

    if args.edge_method == "sam":
        sam_detector = SAMEdgeDetector(
            args.sam_model_type,
            args.device,
            args.sam_points_per_side,
        )

    writer = AsyncImageWriter(min(4, args.num_workers))

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        arrays, paths = dataset.get_batch(i, args.batch_size)

        for arr, path in zip(arrays, paths):
            rel = os.path.relpath(path, args.image_dir).replace("\\", "/")
            stem = os.path.splitext(rel)[0]

            if args.edge_method == "canny":
                edges = canny_edges(arr, args.canny_t1, args.canny_t2)
            else:
                edges = sam_detector.detect_edge(arr)

            out_path = os.path.join(
                args.output_dir, out_subdir, stem + "_edges.png"
            )
            writer.submit(out_path, edges)
            manifest.record(rel)

    writer.close()
    manifest.flush()
    print("Done.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resize", type=int, nargs=2, default=None)

    parser.add_argument("--edge_method", type=str, choices=["canny", "sam"], default="canny")
    parser.add_argument("--canny_t1", type=float, default=255 / 3) # Default values from paper - https://github.com/boyazeng/understand_bias/blob/main/transformations/canny/transform.py
    parser.add_argument("--canny_t2", type=float, default=255.0) # Default values from paper - https://github.com/boyazeng/understand_bias/blob/main/transformations/canny/transform.py

    parser.add_argument("--sam_model_type", choices=["vit_b", "vit_l", "vit_h"], type=str)
    parser.add_argument("--sam_points_per_side", type=int, default=32) # Default values from paper - https://github.com/boyazeng/understand_bias/blob/main/transformations/sam/transform.py
    parser.add_argument("--device", type=str, default="cuda")

    main(parser.parse_args())

# ===========================================================
# EXAMPLE USAGE
# python EdgeDetection.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh --edge_method "canny"

# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --edge_method canny -> This is done as the canny edge detection method along with the SAM method was the primary method used in the paper, however the SAM methods performed slightly better but was much more computationally expensive. Hence canny is preferred.

# OTHER EXAMPLE USAGE
# python EdgeDetection.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh --edge_method "sam" --sam_model_type "vit_l"

# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --edge_method sam -> This is the other model used in the paper
# --sam_model_type vit_l -> This is the model type used in the paper which provided a good balance between performance and computational cost.
# NOTE: This model is quite costly time wise and additionally in the paper it was noted that The classification accuracy on SAM contours (73.2%) is slightly higher than that on Canny edge (71.0%), thus its better to use Canny Edge as opposed to this.

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\canny_sam_test.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\test" --edge_method "canny" --resize 224 224

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\canny_sam_test.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\test" --sam_model_type "vit_l" --edge_method "sam" --resize 224 224