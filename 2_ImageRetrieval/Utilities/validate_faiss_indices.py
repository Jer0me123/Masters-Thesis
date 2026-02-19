import csv
import pickle
from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm


# ============================
# CONFIGURATION
# ============================

BATCH_SIZE = 100_000         # vectors per streaming batch
ATOL = 1e-6                  # float tolerance
REPORT_PATH = "faiss_validation_report.csv"


# ============================
# CORE UTILITIES
# ============================

def load_mapping(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_indices(dir_path: Path):
    """
    Returns:
        dict[str, Path] : index_name -> index_path
    """
    return {
        p.name: p
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".index"
    }


def find_mapping_for_index(index_path: Path):
    """
    Your naming pattern:
        faiss_xxx_IndexFlatIP.index
        faiss_xxx_mapping.pkl
    """
    return index_path.with_name(
        index_path.name.replace("_IndexFlatIP.index", "_mapping.pkl")
    )


def structural_equal(idx_a, idx_b):
    return {
        "type_equal": type(idx_a).__name__ == type(idx_b).__name__,
        "dim_equal": idx_a.d == idx_b.d,
        "ntotal_equal": idx_a.ntotal == idx_b.ntotal,
        "metric_equal": idx_a.metric_type == idx_b.metric_type,
    }


def compare_vectors_streaming(idx_a, idx_b, batch_size=BATCH_SIZE, atol=ATOL):
    """
    Streams vector reconstruction and validates numerical equality.
    Returns:
        (ok: bool, first_bad_offset: int | None, max_diff: float)
    """
    n = idx_a.ntotal
    max_diff_seen = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        Va = np.vstack([idx_a.reconstruct(i) for i in range(start, end)])
        Vb = np.vstack([idx_b.reconstruct(i) for i in range(start, end)])

        diff = np.max(np.abs(Va - Vb))
        max_diff_seen = max(max_diff_seen, float(diff))

        if not np.allclose(Va, Vb, atol=atol):
            return False, start, max_diff_seen

    return True, None, max_diff_seen


# ============================
# MAIN VALIDATOR
# ============================

def validate_directories(dir_a, dir_b):
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    if not dir_a.exists():
        raise FileNotFoundError(dir_a)
    if not dir_b.exists():
        raise FileNotFoundError(dir_b)

    indices_a = collect_indices(dir_a)
    indices_b = collect_indices(dir_b)

    names_a = set(indices_a)
    names_b = set(indices_b)

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    print("\n" + "=" * 80)
    print("FAISS INDEX VALIDATION")
    print("=" * 80)
    print(f"Directory A: {dir_a}")
    print(f"Directory B: {dir_b}")
    print(f"Only in A  : {len(only_a)}")
    print(f"Only in B  : {len(only_b)}")
    print(f"Common     : {len(common)}")

    results = []

    for name in tqdm(common, desc="Validating indices"):
        pa = indices_a[name]
        pb = indices_b[name]

        row = {
            "index_name": name,
            "path_a": str(pa),
            "path_b": str(pb),
            "status": "OK",
        }

        try:
            idx_a = faiss.read_index(str(pa))
            idx_b = faiss.read_index(str(pb))

            # ---------- Structural ----------
            struct = structural_equal(idx_a, idx_b)
            row.update(struct)

            if not all(struct.values()):
                row["status"] = "STRUCT_MISMATCH"
                results.append(row)
                continue

            # ---------- Mapping ----------
            map_a_path = find_mapping_for_index(pa)
            map_b_path = find_mapping_for_index(pb)

            map_a = load_mapping(map_a_path)
            map_b = load_mapping(map_b_path)

            if map_a is None or map_b is None:
                row["mapping_equal"] = False
                row["status"] = "MAPPING_MISSING"
                results.append(row)
                continue

            row["mapping_len_equal"] = len(map_a) == len(map_b)

            if map_a != map_b:
                row["mapping_equal"] = False
                row["status"] = "MAPPING_MISMATCH"
                results.append(row)
                continue

            row["mapping_equal"] = True

            # ---------- Vector Equality ----------
            ok, bad_offset, max_diff = compare_vectors_streaming(idx_a, idx_b)

            row["vector_equal"] = ok
            row["max_vector_diff"] = max_diff
            row["first_bad_offset"] = bad_offset

            if not ok:
                row["status"] = "VECTOR_MISMATCH"
            else:
                row["status"] = "OK"

        except Exception as e:
            row["status"] = "ERROR"
            row["error"] = repr(e)

        results.append(row)

    # ---------- Write CSV ----------
    report_path = Path(REPORT_PATH)
    print(f"\nWriting report → {report_path.resolve()}")

    fieldnames = sorted({k for r in results for k in r.keys()})

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---------- Console Summary ----------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    from collections import Counter
    status_counts = Counter(r["status"] for r in results)

    for status, count in status_counts.items():
        print(f"{status:20s} : {count}")

    if only_a:
        print("\nOnly in A:")
        for n in only_a:
            print("  ", n)

    if only_b:
        print("\nOnly in B:")
        for n in only_b:
            print("  ", n)

    print("\nValidation complete.")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    DIR_A = r"E:\Thesis\image_retrieval_faiss_indices"
    DIR_B = r"G:\Thesis\image_retrieval_faiss_indices"

    validate_directories(DIR_A, DIR_B)
