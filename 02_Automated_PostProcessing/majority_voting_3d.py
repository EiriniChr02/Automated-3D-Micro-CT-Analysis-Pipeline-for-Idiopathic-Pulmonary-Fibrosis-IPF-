import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import convolve
from joblib import Parallel, delayed
import time
import gc

# ══════════════════════════════════════════════════════════════
# SETTINGS & PATHS
# ══════════════════════════════════════════════════════════════
# Define generic paths for GitHub (Replace with local paths when running)
INPUT_DIR = Path("./data/input/classification_results") 
OUTPUT_DIR = Path("./data/output")

# Where the final cleaned TIF will be saved
OUTPUT_FILE = OUTPUT_DIR / "01_majority_voted_volume.tif"

# --- MEMORY OPTIMIZATIONS (IN-PLACE ALLOCATION) ---
CHUNK_SIZE = 20   
MAX_THREADS = 45  
MV_SIZE = 3 # 3x3x3 Kernel size for Majority Voting
# ══════════════════════════════════════════════════════════════

def process_mv_chunk(z_start, z_dim, labels_array, pad):
    """
    TRUE IN-PLACE OVERWRITE: 
    Does not return anything (None). Writes directly to the shared main array in RAM!
    This eliminates memory fragmentation and queuing overhead.
    """
    z_end = min(z_start + CHUNK_SIZE, z_dim)
    z_start_pad = max(0, z_start - pad)
    z_end_pad = min(z_dim, z_end + pad)

    # Read-only view of the current chunk including padding
    sub_vol = labels_array[z_start_pad:z_end_pad]
    
    # Pre-allocation buffers for zero memory fragmentation
    shape = sub_vol.shape
    best_labels_sub = np.zeros(shape, dtype=np.uint8)
    max_density_sub = np.zeros(shape, dtype=np.uint8) 
    
    density_buffer  = np.zeros(shape, dtype=np.uint8)
    mask_buffer     = np.zeros(shape, dtype=np.uint8)
    bool_buffer     = np.zeros(shape, dtype=bool)
    
    kernel = np.ones((MV_SIZE, MV_SIZE, MV_SIZE), dtype=np.uint8)

    # Evaluate density for each class label (0 to 4)
    for l in [0, 1, 2, 3, 4]:
        np.equal(sub_vol, l, out=bool_buffer)
        np.copyto(mask_buffer, 0)
        np.copyto(mask_buffer, 1, where=bool_buffer)
        
        convolve(mask_buffer, kernel, output=density_buffer, mode='constant', cval=0)
        np.greater(density_buffer, max_density_sub, out=bool_buffer)
        
        np.copyto(best_labels_sub, l, casting='unsafe', where=bool_buffer)
        np.copyto(max_density_sub, density_buffer, where=bool_buffer)
        
    valid_start = z_start - z_start_pad
    valid_end = valid_start + (z_end - z_start)
    
    # Write directly to the shared global labels_array (Thread-Safe)
    labels_array[z_start:z_end] = best_labels_sub[valid_start:valid_end]
    
    # Clean local memory explicitly
    del sub_vol, best_labels_sub, max_density_sub, density_buffer, mask_buffer, bool_buffer
    
    if z_end % 500 == 0 or z_end == z_dim:
        print(f"      Completed {z_end}/{z_dim} slices...")

def run_pipeline_part1():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Locating and loading image slices into RAM...")
    slice_files = sorted([f for f in INPUT_DIR.iterdir() if f.suffix.lower() in {'.tif', '.tiff', '.png'}])
    z_lungs = len(slice_files)
    
    if z_lungs == 0:
        print(" [!] ERROR: No images found in the input directory!")
        return

    sample = tifffile.imread(str(slice_files[0]))
    
    # Allocate the entire 3D volume in RAM (Requires High-RAM Workstation)
    labels = np.zeros((z_lungs, sample.shape[0], sample.shape[1]), dtype=sample.dtype)
    
    for z, f in enumerate(slice_files):
        labels[z] = tifffile.imread(str(f))
        if (z+1) % 500 == 0: 
            print(f"  -> Loaded {z+1}/{z_lungs} slices in memory...")

    print(f"\n[2/3] Running 3D Majority Voting with {MAX_THREADS} concurrent cores...")
    print(f"      (True In-Place Overwrite Strategy - Zero Memory Queue)")
    pad = MV_SIZE // 2
    
    # Parallel execution without returning large arrays (sharedmem)
    Parallel(n_jobs=MAX_THREADS, backend='threading', require='sharedmem')(
        delayed(process_mv_chunk)(z, z_lungs, labels, pad) for z in range(0, z_lungs, CHUNK_SIZE)
    )

    print(f"\n[3/3] Saving the massive Volume to disk...")
    tifffile.imwrite(str(OUTPUT_FILE), labels, compression='zlib', bigtiff=True)
    
    del labels; gc.collect()
    
    print(f"\n✓ Success! Process completed in {(time.time() - t_start) / 60:.1f} minutes.")
    print("═" * 60)

if __name__ == "__main__":
    run_pipeline_part1()
