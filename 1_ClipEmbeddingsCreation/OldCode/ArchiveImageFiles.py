import os
import re
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG — EDIT THESE
# ============================================================

SOURCE_DIR = Path(r"G:\Thesis\0000_images")                 # folder with original images
WORK_DIR = Path(r"G:\Thesis\RELaion5BImageArchivesWorkDir")            # temporary shard folders
ARCHIVE_DIR = Path(r"G:\Thesis\RELaion5BImageArchives")       # final .7z files

FILES_PER_SHARD = 250_000

SEVEN_ZIP_EXE = r"C:\Program Files\7-Zip\7z.exe"   # or full path: r"C:\Program Files\7-Zip\7z.exe"

USE_MOVE = False       # False = SAFE (copy); True = destructive move
VERIFY_ARCHIVE = True
DELETE_SHARD_AFTER_ARCHIVE = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ============================================================
# INTERNAL UTILS
# ============================================================

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())
    return p.stdout.decode()


def extract_index(path: Path):
    """
    Extract leading numeric index from filename.
    Example: 000123.jpg → 123
    """
    m = re.match(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


# ============================================================
# STEP 1 — COLLECT & SORT FILES
# ============================================================

def get_sorted_files():
    files = []

    for f in tqdm(SOURCE_DIR.iterdir(), desc="Scanning files", mininterval=0.0):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue

        idx = extract_index(f)
        if idx is None:
            continue

        files.append((idx, f))

        if len(files) >= 250_000:
            break

    files.sort(key=lambda x: x[0])
    return files


# ============================================================
# STEP 2 — CREATE SHARDS (RESUME SAFE)
# ============================================================

def write_shard(shard_files):
    shard_start = shard_files[0][0]
    shard_end = shard_files[-1][0]
    shard_name = f"shard_{shard_start}_{shard_end}"
    shard_dir = WORK_DIR / shard_name

    if shard_dir.exists():
        print(f"Shard exists, skipping: {shard_name}")
        return

    shard_dir.mkdir(parents=True)

    for _, src in shard_files:
        dst = shard_dir / src.name
        if dst.exists():
            continue

        if USE_MOVE:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)


def create_shards():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    files = get_sorted_files()
    shard_files = []

    for idx, path in tqdm(files, desc="Sharding files", mininterval=0.0):
        shard_files.append((idx, path))

        if len(shard_files) >= FILES_PER_SHARD:
            write_shard(shard_files)
            shard_files = []

    if shard_files:
        write_shard(shard_files)


# ============================================================
# STEP 3 — ARCHIVE SHARDS
# ============================================================

def archive_shards():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    shards = sorted(WORK_DIR.glob("shard_*"))

    for shard in shards:
        archive_path = ARCHIVE_DIR / f"{shard.name}.7z"

        if archive_path.exists():
            try:
                run([SEVEN_ZIP_EXE, "t", str(archive_path)])
                print(f"Verified existing archive, skipping {archive_path.name}")
                continue
            except RuntimeError:
                print(f"Corrupt archive found, re-archiving {archive_path.name}")
                archive_path.unlink()


        print(f"Archiving {shard.name}")

        # No compression but reduced to several larger singualr files instead for better maintainability.
        run([
            SEVEN_ZIP_EXE,
            "a",
            "-t7z",
            # "-m0=lzma2",
            "-mx=0", # "-mx=9"
            "-ms=on", #"-ms=on",
            str(archive_path),
            str(shard / "*"),
        ])

        if VERIFY_ARCHIVE:
            run([SEVEN_ZIP_EXE, "t", str(archive_path)])

        if DELETE_SHARD_AFTER_ARCHIVE:
            shutil.rmtree(shard)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=== Creating shards ===")
    create_shards()

    print("\n=== Archiving shards ===")
    archive_shards()

    print("\nDONE.")

# python ArchiveImageFiles.py