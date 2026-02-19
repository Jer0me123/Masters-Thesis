import ctypes
import ctypes.wintypes as wt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import json
import os
import sys

# --- Windows API Setup ---
CopyFileExW = ctypes.windll.kernel32.CopyFileExW
COPY_FILE_RESTARTABLE = 0x00000002 
LPBOOL = ctypes.POINTER(wt.BOOL)

CopyFileExW.argtypes = [
    wt.LPCWSTR, wt.LPCWSTR, wt.LPVOID, wt.LPVOID, LPBOOL, wt.DWORD
]
CopyFileExW.restype = wt.BOOL

def copy_results_global(all_results, image_dir, output_dir, max_workers=4):
    """
    Inner function: Handles the copying logic.
    """
    src_root = Path(image_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Cap workers to avoid disk thrashing
    max_workers = max(1, min(max_workers, 16))

    def resolve_src(r):
        raw = Path(r["image_path"])
        if raw.is_absolute(): return raw
        # Adjust this path joining logic if your folder structure differs
        return src_root / raw.parent.parent / f"{r['group_id']}_images" / raw.name

    def win_copy(src: Path, dst: Path):
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cancel_flag = wt.BOOL(False)
        
        # Windows Kernel Copy
        ok = CopyFileExW(str(src), str(tmp), None, None, ctypes.byref(cancel_flag), COPY_FILE_RESTARTABLE)
        
        if not ok: 
            # Get Windows Error Code for better debugging
            err = ctypes.GetLastError()
            raise OSError(f"CopyFileExW failed for {src} (Error Code: {err})")
        
        # Atomic rename
        if dst.exists():
            dst.unlink()
        tmp.replace(dst)

    # Prepare items
    items = []
    for entry in all_results:
        # Sanitize prompt for folder name
        safe_prompt = entry["prompt"].replace(" ", "_").replace("/", "_").replace("\\", "_")
        pdir = out_root / safe_prompt
        
        for r in entry["results"][:500_000]:
            items.append((pdir, r))

    skipped_count = 0
    futures = []

    # Filter existing files
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for pdir, r in items:
            src = resolve_src(r)
            dst = pdir / f"{r['score']:.3f}_{r['group_id']}_{src.name}"
            
            should_copy = True
            if dst.exists():
                try:
                    # Size check avoids re-copying valid files
                    if src.stat().st_size == dst.stat().st_size:
                        should_copy = False
                        skipped_count += 1
                except OSError: 
                    pass # If src missing, let win_copy fail loudly later

            if should_copy:
                futures.append(ex.submit(win_copy, src, dst))

        # IF there is work to do, show the INNER progress bar
        if futures:
            # position=1 ensures it renders BELOW the main bar in terminal
            # leave=False ensures it clears after the batch is done
            for f in tqdm(as_completed(futures), total=len(futures), desc="  Batch Copy", unit="img", position=1, leave=False, colour='green'):
                try:
                    f.result()
                except Exception as e:
                    tqdm.write(f"Error: {e}") 

def copy_results_from_jsonl(jsonl_path, image_dir, output_dir, max_workers=4, batch_size=10):
    """
    Outer function: Stream JSONL with a persistent main progress bar.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        return

    total_bytes = os.path.getsize(jsonl_path)
    batch = []
    
    print(f"Starting processing of {jsonl_path}...")
    print(f"Output Directory: {output_dir}")

    # position=0 is the TOP bar. leave=True keeps it visible till the end.
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Total Progress", position=0, leave=True, colour='blue') as pbar:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                # Update main bar by bytes read
                pbar.update(len(line.encode('utf-8'))) 
                
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError:
                    tqdm.write("Skipped corrupt line")
                    continue

                if len(batch) >= batch_size:
                    copy_results_global(batch, image_dir, output_dir, max_workers)
                    batch.clear()

            # Final flush
            if batch:
                copy_results_global(batch, image_dir, output_dir, max_workers)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # You can change these paths here or use sys.argv to pass them in
    
    JSONL_PATH = r"G:\Thesis\ImageRetrieval\Professions_125k\retrieval_results_batchsize_10.jsonl"
    IMAGE_ROOT = r"G:\Thesis"
    OUTPUT_DIR = r"G:\Thesis\ImageRetrieval\Professions_125k_test"
    
    # 32 workers is aggressive but fine for SSDs. Lower to 8-16 for HDDs.
    MAX_WORKERS = 32
    BATCH_SIZE = 10 

    copy_results_from_jsonl(
        jsonl_path=JSONL_PATH,
        image_dir=IMAGE_ROOT,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE
    )