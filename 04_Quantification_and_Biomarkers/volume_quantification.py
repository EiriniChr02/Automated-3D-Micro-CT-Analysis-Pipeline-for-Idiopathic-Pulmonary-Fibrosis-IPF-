import numpy as np
import tifffile
from pathlib import Path
import time
import os

# Define path
# The Path can be a folder of 2D slices or a single 3D .tif file
INPUT_PATH = Path("./data/output/03_airway_closed_volume.tif")

# Voxel size in μm from the Micro-CT scanner
# (Necessary to convert pixel counts to cubic millimeters - mm³)
VOXEL_SIZE_UM = 5.5

# Tissue Classification Labels dictionary
LABELS_DICT = {
    0: "Background / Air",
    1: "Healthy Tissue",
    2: "Fibrosis",
    3: "Airway Walls",
    4: "Mediastinum"
}

def quantify_all_classes():
    t_start = time.time()
    print("═" * 60)
    print("  3D QUANTIFICATION OF ALL CLASSES (100% RAM-SAFE)")
    print("═" * 60)

    if not INPUT_PATH.exists():
        print(f" [!] ERROR: The file or folder {INPUT_PATH.name} was not found!")
        return

    # Calculation of the volume of a single voxel in mm³
    voxel_vol_mm3 = (VOXEL_SIZE_UM / 1000.0) ** 3
    print(f"\n[1/3] Analysis Parameters:")
    print(f"  -> Voxel Size: {VOXEL_SIZE_UM} μm")
    print(f"  -> Volume of 1 Voxel: {voxel_vol_mm3:.8f} mm³")

    # Array to keep the sum of all pixels (max label = 4, so size 5 is needed)
    total_counts = np.zeros(5, dtype=np.int64)

    print(f"\n[2/3] Reading and Calculating voxels (Slice-by-Slice)...")
    
    is_dir = INPUT_PATH.is_dir()
    
    try:
        if is_dir:
            # FOLDER MODE
            valid_exts = {'.tif', '.tiff', '.png'}
            slice_files = sorted([f for f in INPUT_PATH.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
            
            z_dim = len(slice_files)
            if z_dim == 0:
                print(f" [!] ERROR: No TIF Files found in the folder {INPUT_PATH.name}!")
                return
                
            print(f"  Folder Mode: Found {z_dim} slices.")
            print("  Scanning (without RAM consumption)...\n")

            for z, f_path in enumerate(slice_files):
                slice_data = tifffile.imread(str(f_path))
                counts = np.bincount(slice_data.ravel(), minlength=5)[:5]
                total_counts += counts

                if z % 500 == 0 and z > 0:
                    print(f"      Checked {z}/{z_dim} slices...")
                    
        else:
            # FILE MODE (3D TIF)
            with tifffile.TiffFile(str(INPUT_PATH)) as tif_in:
                z_dim = len(tif_in.pages)
                print(f"  File Mode: Found {z_dim} slices.")
                print("  Scanning (without RAM consumption)...\n")
                
                for z, page in enumerate(tif_in.pages):
                    slice_data = page.asarray()
                    counts = np.bincount(slice_data.ravel(), minlength=5)[:5]
                    total_counts += counts

                    if z % 500 == 0 and z > 0:
                        print(f"      Checked {z}/{z_dim} slices...")

    except Exception as e:
        print(f" [!] Error while reading: {e}")
        return

    # Statistics Calculation
    print("\n[3/3] Calculating Volumes and Percentages...")
    
    # Total lung tissue ignores the air (Background - Label 0)
    total_tissue_voxels = np.sum(total_counts[1:]) 
    
    print("\n" + "═"*50)
    print("  RESULTS")
    print("═" * 50)
    
    for label_val, label_name in LABELS_DICT.items():
        count = total_counts[label_val]
        vol_mm3 = count * voxel_vol_mm3
        
        # Percentage of the total volume of the entire scan
        total_image_pct = (count / np.sum(total_counts)) * 100 if np.sum(total_counts) > 0 else 0
        
        # Percentage of the lung tissue specifically (Labels 1, 2, 3, 4 only)
        tissue_pct = (count / total_tissue_voxels) * 100 if label_val > 0 and total_tissue_voxels > 0 else 0.0
        
        # Output formatting
        if label_val == 0:
            print(f"  {label_name:<18}: {vol_mm3:>8.2f} mm³ | Image: {total_image_pct:>5.1f}%")
        else:
            print(f"  {label_name:<18}: {vol_mm3:>8.2f} mm³ | Tissue: {tissue_pct:>5.1f}%")

    print("-" * 50)
    
    # Clinical Statistics (Tissue Only)
    if total_tissue_voxels > 0:
        print(f"\n  Clinical statistics of the tissue (Excluding Background Air):")
        print(f"     Total Tissue: {(total_tissue_voxels * voxel_vol_mm3):.2f} mm³")
        
        # Dynamic percentage calculation for each tissue class
        for label_val, label_name in LABELS_DICT.items():
            if label_val == 0:
                continue
            pct = (total_counts[label_val] / total_tissue_voxels) * 100
            print(f"     -> {label_name:<14}: {pct:>5.1f} %")

    print(f"\n✓ Successful calculations in {(time.time() - t_start):.1f} seconds!")
    print("═" * 60)

if __name__ == "__main__":
    quantify_all_classes()
