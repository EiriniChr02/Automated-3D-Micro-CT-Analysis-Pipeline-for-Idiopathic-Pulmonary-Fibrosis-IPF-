import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage
import time
import gc

# ══════════════════════════════════════════════════════════════
# SETTINGS & PATHS
# ══════════════════════════════════════════════════════════════
# Define generic paths for GitHub
INPUT_DIR = Path("./data/output") 
OUTPUT_DIR = Path("./data/output")

# Input is the output from the Majority Voting step
INPUT_FILE = INPUT_DIR / "01_majority_voted_volume.tif"
OUTPUT_FILE = OUTPUT_DIR / "02_dust_cleaned_volume.tif"

# Tissue Classification Labels
LABEL_BG       = 0
LABEL_HEALTHY  = 1
LABEL_FIBROSIS = 2
LABEL_WALLS    = 3
LABEL_MEDIAST  = 4

# Voxel Thresholds for 3D Connected-Component Analysis
MIN_MEDIAST_3D = 200_000  
MIN_DUST_3D    = 1_000    
MIN_TISSUE_3D  = 100_000  

# RAM-Safe Chunking Parameters
CHUNK_Z = 120   
PAD_Z   = 40    
# ══════════════════════════════════════════════════════════════

def run_dust_cleanup():
    t_start = time.time()
    print("═" * 60)
    print("  VECTORIZED SEQUENTIAL DUST CLEANUP (MAX SPEED & SAFE RAM)")
    print("═" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/3] Loading the entire 3D volume...\n      {INPUT_FILE.name}")
    labels = tifffile.imread(str(INPUT_FILE))
    z_dim = labels.shape[0]
    print(f"  -> Shape: {labels.shape}")

    print(f"\n[2/3] Executing Chunk-by-Chunk Cleanup (Vectorized EDT)...")
    
    t_proc = time.time()
    
    # Process 1 Chunk at a time, but calculations inside are MASSIVELY parallelized via Scipy
    for z_start in range(0, z_dim, CHUNK_Z):
        z_end = min(z_start + CHUNK_Z, z_dim)
        z_s_pad = max(0, z_start - PAD_Z)
        z_e_pad = min(z_dim, z_end + PAD_Z)
        
        print(f"  > Processing Chunk [{z_start} - {z_end}]...")
        
        sub_vol = labels[z_s_pad:z_e_pad].copy()
        
        # Define sequential cleanup steps
        steps = [
            (LABEL_MEDIAST, MIN_MEDIAST_3D, True, "Mediastinum"),
            (LABEL_FIBROSIS, MIN_DUST_3D, True, "Fibrosis"),
            (LABEL_WALLS, MIN_DUST_3D, True, "Airway Walls"),
            ("TISSUE", MIN_TISSUE_3D, False, "Floating Tissue") # False = simple deletion
        ]
        
        for target_label, min_size, smart_replace, name in steps:
            if target_label == "TISSUE":
                mask = (sub_vol != LABEL_BG)
            else:
                mask = (sub_vol == target_label)
                
            if not np.any(mask):
                continue

            # 3D Connected-Component Labeling
            labeled, num_features = ndimage.label(mask)
            if num_features == 0:
                continue
                
            # ── SAFE BINCOUNT (Prevents memory overflow on large numbers of features) ──
            sizes = np.zeros(num_features + 1, dtype=np.int64)
            for z_slice in labeled:
                counts = np.bincount(z_slice.ravel())
                sizes[:len(counts)] += counts
                
            # Boundary Protection: Ignore objects touching the Z boundaries of the chunk
            boundary_labels = set(np.unique(labeled[0, :, :]))
            boundary_labels.update(np.unique(labeled[-1, :, :]))
            boundary_labels.discard(0)
            
            if boundary_labels:
                sizes[list(boundary_labels)] = 1_000_000_000 # Make them artificially large
                
            small_ids = np.where((sizes > 0) & (sizes < min_size))[0]
            
            if len(small_ids) > 0:
                print(f"    - Identified {len(small_ids):,} small objects [{name}]. Performing immediate replacement...")
                
                # Create a mask with ALL the dust/noise combined
                is_small = np.zeros(num_features + 1, dtype=bool)
                is_small[small_ids] = True
                small_mask = is_small[labeled]
                
                if smart_replace:
                    # ── VECTORIZED EDT (LIGHTNING-FAST NEIGHBOR SEARCH) ──
                    # Finds nearest valid neighbor for ALL isolated objects simultaneously!
                    indices = ndimage.distance_transform_edt(small_mask, return_distances=False, return_indices=True)
                    
                    nearest_z = indices[0][small_mask]
                    nearest_y = indices[1][small_mask]
                    nearest_x = indices[2][small_mask]
                    
                    # Color replacement (Context-Aware smart filling)
                    sub_vol[small_mask] = sub_vol[nearest_z, nearest_y, nearest_x]
                    
                    del indices, nearest_z, nearest_y, nearest_x
                else:
                    # Simple deletion for artifacts completely outside the lung
                    sub_vol[small_mask] = LABEL_BG
                    
                del is_small, small_mask
                
            del labeled, sizes, mask
            gc.collect()
            
        # Write back to the main array
        valid_start = z_start - z_s_pad
        valid_end = valid_start + (z_end - z_start)
        labels[z_start:z_end] = sub_vol[valid_start:valid_end]
        
        del sub_vol
        gc.collect()
        
    print(f"\n  -> Processing completed in {(time.time() - t_proc) / 60:.1f} minutes!")

    print(f"\n[3/3] Saving the cleaned TIF to disk...\n      {OUTPUT_FILE.name}")
    tifffile.imwrite(str(OUTPUT_FILE), labels, compression='zlib', bigtiff=True)
    
    del labels; gc.collect()
    
    print(f"\n✓ Success! Total time: {(time.time() - t_start) / 60:.1f} minutes.")
    print("═" * 60)

if __name__ == "__main__":
    run_dust_cleanup()
