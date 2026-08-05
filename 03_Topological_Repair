import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import binary_closing, generate_binary_structure
from joblib import Parallel, delayed
import time
import gc

# ══════════════════════════════════════════════════════════════
# SETTINGS & PATHS (OPTIMIZED FOR HIGH-END WORKSTATIONS)
# ══════════════════════════════════════════════════════════════
# Define generic paths for GitHub
INPUT_DIR = Path("./data/output") 
OUTPUT_DIR = Path("./data/output")

# Input is the dust-cleaned volume from the previous post-processing step
INPUT_FILE = INPUT_DIR / "02_dust_cleaned_volume.tif"
OUTPUT_FILE  = OUTPUT_DIR / "03_airway_closed_volume.tif"

LABEL_WALLS = 3

# Memory Constraints & Algorithmic Parameters
CHUNK_SIZE         = 25  # Reduced to avoid ArrayMemoryError & RAM fragmentation
MAX_THREADS        = 16  
CLOSING_ITERATIONS = 4   # Morphological iterations (4-6) to bridge larger structural gaps
# ══════════════════════════════════════════════════════════════

def process_closing_chunk(z_start, z_dim, labels_array, pad):
    z_end = min(z_start + CHUNK_SIZE, z_dim)
    z_start_pad = max(0, z_start - pad)
    z_end_pad = min(z_dim, z_end + pad)

    sub_vol = labels_array[z_start_pad:z_end_pad]
    airways_sub = (sub_vol == LABEL_WALLS)
    
    # 26-connectivity: Connects pixels diagonally in all 3D directions.
    # This makes the reconstructed airway wall much more compact, natural, and watertight.
    struct = generate_binary_structure(3, 3)
    
    # Apply 3D Binary Closing
    closed_sub = binary_closing(airways_sub, structure=struct, iterations=CLOSING_ITERATIONS)
    
    # Isolate the newly generated bridges/walls
    new_bridges_sub = closed_sub & ~airways_sub
    
    # Force replacement: If the morphological algorithm determines a voxel belongs 
    # inside the airway wall structure, we convert it to LABEL_WALLS regardless 
    # of the underlying misclassified tissue (e.g., fibrosis, blood vessels).
    pixels_to_change = new_bridges_sub

    valid_start = z_start - z_start_pad
    valid_end = valid_start + (z_end - z_start)
    pixels_to_change_valid = pixels_to_change[valid_start:valid_end]

    # In-place overwrite (Thread-safe assignment directly to the global array)
    target_slice = labels_array[z_start:z_end]
    target_slice[pixels_to_change_valid] = LABEL_WALLS

    if z_end % 500 == 0 or z_end == z_dim:
        print(f"      Completed {z_end}/{z_dim} slices...")

def run_airway_closing():
    t_start = time.time()
    print("═" * 60)
    print("  TOPOLOGICAL REPAIR: AGGRESSIVE 3D AIRWAY CLOSING")
    print("═" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── 1. LOADING ──
    print(f"\n[1/3] Loading data into RAM from:\n      {INPUT_FILE.name}...")
    try:
        labels = tifffile.imread(str(INPUT_FILE))
        z_lungs = labels.shape[0]
        print(f"  -> Shape: {labels.shape}")
    except Exception as e:
        print(f" [!] Error loading file: {e}")
        return

    # ── 2. PROCESSING ──
    print(f"\n[2/3] Airway 3D closing (Enhanced 3D/26-connectivity)...")
    print(f"      Executing with {MAX_THREADS} concurrent cores ({CLOSING_ITERATIONS} iterations)...")
    
    pad = CLOSING_ITERATIONS
    
    # Parallel execution using shared memory to prevent RAM duplication
    Parallel(n_jobs=MAX_THREADS, backend='threading', require='sharedmem')(
        delayed(process_closing_chunk)(z, z_lungs, labels, pad) for z in range(0, z_lungs, CHUNK_SIZE)
    )
    gc.collect()

    # ── 3. SAVING ──
    print(f"\n[3/3] Saving the massive TIF to disk...\n      {OUTPUT_FILE.name}")
    tifffile.imwrite(str(OUTPUT_FILE), labels, compression='zlib', bigtiff=True)
    
    del labels; gc.collect()
    
    print(f"\n✓ Success! Process completed in {(time.time() - t_start) / 60:.1f} minutes.")
    print("═" * 60)

if __name__ == "__main__":
    run_airway_closing()
