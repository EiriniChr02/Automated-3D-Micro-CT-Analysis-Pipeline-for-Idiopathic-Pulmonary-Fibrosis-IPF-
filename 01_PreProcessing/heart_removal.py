import numpy as np
import tifffile
from pathlib import Path
import time
import gc

# ── PARAMETERS ────────────────────────────────────────────────
# Define generic paths (Replace with local paths when running)
INPUT_DIR  = Path("./data/input") 
OUTPUT_DIR = Path("./data/output")

# Create output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. The original/large TIF (e.g., after Histogram Matching)
INPUT_ORIGINAL = INPUT_DIR / "raw_micro_ct_scan.tif"

# 2. The small mask (TIF) obtained from nnInteractive / AI segmentation
INPUT_MASK_SMALL = INPUT_DIR / "low_res_heart_mask.tif"

# 3. Where the final clean TIF will be saved
OUTPUT_FINAL = OUTPUT_DIR / "clean_lung_scan.tif"

# 4. Prior knowledge/information:
UPSAMPLE_FACTOR = 4  # Factor by which the low-res mask should be upsampled in XY
# ──────────────────────────────────────────────────────────────

def remove_heart_memory_safe():
    t_start = time.time()
    print("═"*60)
    print("  HEART REMOVAL VIA ABSOLUTE STREAMING (TRUE SLICE-BY-SLICE)")
    print("═"*60)

    # 1. Load the Mask and immediate conversion to Boolean (1-bit)
    print(f"\n[1/3] Loading the low-res mask (Conversion to 1-bit)...")
    try:
        small_mask = tifffile.imread(str(INPUT_MASK_SMALL)) > 0
        z_mask_dim = small_mask.shape[0]
        print(f"  -> Small Mask Dimensions: {small_mask.shape}")
    except Exception as e:
        print(f" [!] Mask reading error: {e}")
        return

    # 2. Safe Processing with Streaming (1 Slice at a time)
    print(f"\n[2/3] Reading, Processing & Saving slice-by-slice (Streaming)...")
    print("  (Zero additional memory consumption!)")
    
    try:
        # Open the file for reading (WITHOUT loading it entirely to RAM) and the output file
        with tifffile.TiffFile(str(INPUT_ORIGINAL)) as tif_in, \
             tifffile.TiffWriter(str(OUTPUT_FINAL), bigtiff=True) as tif_out:
             
            z_dim_original = len(tif_in.pages)
            print(f"  -> The original TIF has a total of {z_dim_original} slices (Z).")
            print(" Starting Streaming...")

            first_slice = True
            
            # Read, process, and write exactly one slice at a time
            for z, page in enumerate(tif_in.pages):
                # Load ONLY the specific 2D slice into RAM
                img_slice = page.asarray()
                
                # Apply the mask from the beginning, as long as we haven't exceeded the mask's Z dimension
                if z < z_mask_dim:
                    slice_small = small_mask[z]
                    
                    # Upsampling ONLY the specific 2D mask slice
                    slice_large = np.repeat(np.repeat(slice_small, UPSAMPLE_FACTOR, axis=0), UPSAMPLE_FACTOR, axis=1)
                    
                    y_dim, x_dim = img_slice.shape
                    valid_mask = slice_large[:y_dim, :x_dim]
                    
                    # Zero out the heart region voxels
                    img_slice[:valid_mask.shape[0], :valid_mask.shape[1]][valid_mask] = 0
                
                # Write the processed slice directly to the final file on disk
                if first_slice:
                    tif_out.write(img_slice, compression='zlib', metadata={'axes': 'ZYX'})
                    first_slice = False
                else:
                    tif_out.write(img_slice, compression='zlib')

                # Update progress
                if z % 200 == 0 and z > 0:
                    print(f"      Completed {z}/{z_dim_original} slices...")

        del small_mask; gc.collect()
        
    except Exception as e:
        print(f" [!] Error during Streaming: {e}")
        return

    print(f"\n[3/3] File successfully written to disk!")
    print(f"Completed with memory safety in {(time.time() - t_start) / 60:.1f} minutes!")
    print(f"  The final file is: {OUTPUT_FINAL.name}")
    print("═"*60)

if __name__ == "__main__":
    remove_heart_memory_safe()
