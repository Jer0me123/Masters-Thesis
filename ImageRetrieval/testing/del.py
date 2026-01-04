import os
import subprocess
from pathlib import Path
import time
import concurrent.futures
from tqdm import tqdm
import multiprocessing

def delete_single_directory(dir_path):
    """Delete a single directory using Windows rmdir command"""
    dir_path = str(dir_path)
    start = time.time()
    
    try:
        result = subprocess.run(
            ['cmd', '/c', 'rmdir', '/S', '/Q', dir_path],
            capture_output=True,
            text=True,
            timeout=None
        )
        elapsed = time.time() - start
        return {
            'path': dir_path,
            'success': result.returncode == 0 or not Path(dir_path).exists(),
            'elapsed': elapsed,
            'error': result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'path': dir_path,
            'success': False,
            'elapsed': elapsed,
            'error': str(e)
        }

def delete_directory_parallel(directory: str, max_workers: int = None):
    """
    Parallel deletion using multiple processes.
    Each subdirectory gets deleted by a separate process using rmdir.
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 processes
    
    print(f"╔{'═'*70}╗")
    print(f"║ PARALLEL NUCLEAR DELETION - Multi-Process rmdir{' '*22}║")
    print(f"╚{'═'*70}╝\n")
    
    print(f"Target: {directory}")
    print(f"Workers: {max_workers} parallel processes\n")
    
    print(f"WARNING: This will delete ALL contents recursively.")
    print(f"Press Ctrl+C within 5 seconds to cancel...\n")
    
    try:
        for i in range(5, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    print("\n" + "="*70)
    print("Scanning for subdirectories...")
    
    # Find all immediate subdirectories
    subdirs = []
    root_files_exist = False
    
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    root_files_exist = True
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return
    
    print(f"Found {len(subdirs)} subdirectories to delete in parallel")
    
    if root_files_exist:
        print("Note: Root directory contains files (will be deleted last)")
    
    print("="*70 + "\n")
    
    overall_start = time.time()
    results = []
    
    if subdirs:
        print(f"Deleting {len(subdirs)} subdirectories using {max_workers} parallel processes...")
        print("(Each subdirectory deletion shows no progress but is working)\n")
        
        # Use ProcessPoolExecutor for true parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all subdirectory deletions
            future_to_dir = {executor.submit(delete_single_directory, d): d for d in subdirs}
            
            # Use tqdm to show completion progress
            with tqdm(total=len(subdirs), desc="Subdirectories", unit="dir") as pbar:
                for future in concurrent.futures.as_completed(future_to_dir):
                    result = future.result()
                    results.append(result)
                    
                    status = "✓" if result['success'] else "✗"
                    dir_name = Path(result['path']).name
                    pbar.set_postfix_str(f"{status} {dir_name} ({result['elapsed']:.1f}s)")
                    pbar.update(1)
        
        print()
    
    # Delete remaining root files if any
    if root_files_exist or subdirs:
        print("Cleaning up root directory...")
        root_result = delete_single_directory(directory)
        results.append(root_result)
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print("\n" + "="*70)
    print("DELETION SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n✓ Successful: {successful}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
        print("\nFailed directories:")
        for r in results:
            if not r['success']:
                print(f"  - {Path(r['path']).name}: {r['error']}")
    
    print(f"\nTotal time: {overall_elapsed:.1f} seconds")
    print(f"Average time per directory: {overall_elapsed/max(1, len(subdirs)):.1f} seconds")
    
    # Check if main directory is gone
    if not directory.exists():
        print(f"\n✓ Main directory successfully removed!")
    else:
        print(f"\n⚠ Main directory still exists - may contain protected files")
    
    print("="*70 + "\n")

def main():
    target = r'G:\Thesis\ImageRetrieval\Professions_20k_DELETE'
    
    print("\n" + "="*72)
    print(" PARALLEL FAST DIRECTORY DELETION TOOL")
    print("="*72 + "\n")
    
    print("This tool deletes multiple subdirectories simultaneously")
    print("using Windows optimized commands in parallel processes.\n")
    
    # Auto-detect optimal workers (but let user override)
    cpu_count = os.cpu_count() or 4
    workers = min(cpu_count, 8)
    
    print(f"Detected {cpu_count} CPU cores, using {workers} parallel workers\n")
    
    delete_directory_parallel(target, max_workers=workers)
    
    print("="*72)
    print(" OPERATION COMPLETE")
    print("="*72 + "\n")

if __name__ == "__main__":
    main()