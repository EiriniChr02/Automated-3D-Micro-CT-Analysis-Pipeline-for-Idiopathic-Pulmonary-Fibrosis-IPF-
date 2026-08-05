import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import distance_transform_edt, convolve
from skimage.morphology import skeletonize
from joblib import Parallel, delayed
import json
import time
import gc

# ══════════════════════════════════════════════════════════════
# SETTINGS (FOR HIGH-RAM WORKSTATIONS & LARGE MICRO-CT STACKS)
# ══════════════════════════════════════════════════════════════
# Generic paths for GitHub
INPUT_DIR = Path("./data/output")
REPORT_DIR = Path("./reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs & Outputs
TREE_FILE       = INPUT_DIR / "04_solid_lumen_tree.tif"       # The solid lumen tree (Result of Hole Filling)
LABELS_FILE     = INPUT_DIR / "03_airway_closed_volume.tif"   # The file containing airway walls
REPORT_FILE     = REPORT_DIR / "biomarkers_report.json"

LABEL_WALL      = 3
VOXEL_SIZE_UM   = 5.5
PERCENTILE_CUTOFF = 95

# RAM-Safe Chunking Parameters for Distance Transforms
CHUNK_Z     = 100  # Number of Z-slices to process at once
PAD_Z       = 40   # Padding (in voxels) to ensure accurate 3D Distance Transform across chunk borders
MAX_THREADS = 16   
# ══════════════════════════════════════════════════════════════

def process_topology_chunk(z_start, chunk_z, z_dim, skeleton):
    """Calculates bifurcations and endpoints safely within a chunk."""
    pad = 1 # Only 1 voxel padding needed for 3x3x3 convolution
    z_end = min(z_start + chunk_z, z_dim)
    z_s_pad = max(0, z_start - pad)
    z_e_pad = min(z_dim, z_end + pad)

    sub_skel = skeleton[z_s_pad:z_e_pad].astype(np.uint8)
    
    kernel = np.ones((3,3,3), dtype=np.uint8)
    kernel[1,1,1] = 0
    neigh = convolve(sub_skel, kernel, mode='constant', cval=0)

    # Remove padding to only count the valid chunk zone
    v_s = z_start - z_s_pad
    v_e = v_s + (z_end - z_start)

    valid_skel = sub_skel[v_s:v_e] > 0
    valid_neigh = neigh[v_s:v_e]

    n_bif = np.sum(valid_skel & (valid_neigh > 2))
    n_end = np.sum(valid_skel & (valid_neigh == 1))

    return int(n_bif), int(n_end)

def process_metrics_chunk(z_start, chunk_z, z_dim, lumen_mask, labels, skeleton, pad):
    """Calculates Wall Thickness and Lumen Diameter using RAM-Safe Chunked EDT."""
    z_end = min(z_start + chunk_z, z_dim)
    z_s_pad = max(0, z_start - pad)
    z_e_pad = min(z_dim, z_end + pad)

    sub_lumen  = lumen_mask[z_s_pad:z_e_pad]
    sub_labels = labels[z_s_pad:z_e_pad]
    sub_skel   = skeleton[z_s_pad:z_e_pad]

    # EDT for Diameter (Distance from lumen center to wall)
    dist_skel = distance_transform_edt(sub_lumen).astype(np.float32)

    # EDT for WT Part 1 (Distance from lumen to airway wall)
    dist_l = distance_transform_edt(~sub_lumen).astype(np.float32)

    # EDT for WT Part 2 (Distance from surrounding tissue to airway wall)
    sub_wall = (sub_labels == LABEL_WALL)
    tissue_m = ~sub_lumen & ~sub_wall
    dist_e = distance_transform_edt(~tissue_m).astype(np.float32)

    # Remove padding to extract true valid values avoiding border artifacts
    v_s = z_start - z_s_pad
    v_e = v_s + (z_end - z_start)

    valid_wall = sub_wall[v_s:v_e]
    valid_skel = sub_skel[v_s:v_e] > 0

    # Extract distances only at the specific target voxels (Walls and Skeleton)
    wt_part1_vals = dist_l[v_s:v_e][valid_wall]
    wt_part2_vals = dist_e[v_s:v_e][valid_wall]
    skel_diams    = dist_skel[v_s:v_e][valid_skel]

    return wt_part1_vals, wt_part2_vals, skel_diams

def run_biomarker_pipeline():
    t_start = time.time()
    print("═" * 60)
    print("  RAM-SAFE BIOMARKERS EXTRACTION (WT & D RATIO)")
    print("═" * 60)

    # ── 1. Data Loading ──────────────────────────────────────────
    print(f"\n[1/4] Loading Large 3D Volumes into RAM...")
    tree = tifffile.imread(str(TREE_FILE))
    labels = tifffile.imread(str(LABELS_FILE))
    
    # Keep as bool/uint8 to minimize memory footprint
    lumen_mask = tree > 0
    air_uint8 = lumen_mask.astype(np.uint8)
    z_dim = tree.shape[0]
    
    print(f"  -> Volume Shape: {tree.shape}")
    del tree; gc.collect()

    # ── 2. Global Skeletonization ────────────────────────────────
    # Skeletonization MUST be done globally to avoid disconnected network branches
    print(f"\n[2/4] Extracting 3D Skeleton (Centerline)...")
    t0 = time.time()
    skeleton = skeletonize(air_uint8)
    print(f"  -> Completed in {(time.time() - t0):.1f}s")
    del air_uint8; gc.collect()

    # ── 3. Chunked Topology Analysis (Bifurcations) ──────────────
    print(f"\n[3/4] Calculating Network Topology (Chunked)...")
    starts = list(range(0, z_dim, CHUNK_Z))
    
    # Shared memory multiprocessing (Read-only data access)
    topology_results = Parallel(n_jobs=MAX_THREADS, require='sharedmem')(
        delayed(process_topology_chunk)(z, CHUNK_Z, z_dim, skeleton) for z in starts
    )
    
    n_bif = sum([res[0] for res in topology_results])
    n_end = sum([res[1] for res in topology_results])
    
    del topology_results; gc.collect()

    # ── 4. Chunked Distance Transforms (WT & D) ──────────────────
    print(f"\n[4/4] Calculating Wall Thickness & Lumen Diameter (Chunked EDT)...")
    
    metrics_results = Parallel(n_jobs=MAX_THREADS, require='sharedmem')(
        delayed(process_metrics_chunk)(z, CHUNK_Z, z_dim, lumen_mask, labels, skeleton, PAD_Z) 
        for z in starts
    )

    # Concatenate results from all chunks
    wt_part1 = np.concatenate([res[0] for res in metrics_results])
    wt_part2 = np.concatenate([res[1] for res in metrics_results])
    skel_diams = np.concatenate([res[2] for res in metrics_results])
    
    del metrics_results, lumen_mask, labels, skeleton; gc.collect()

    # Convert pixel distances to micrometers (μm)
    wt_um = (wt_part1 + wt_part2) * VOXEL_SIZE_UM
    skel_diams_um = skel_diams * 2 * VOXEL_SIZE_UM
    del wt_part1, wt_part2; gc.collect()

    # ── 5. Filtering & Biomarker Extraction ──────────────────────
    print(f"\n[Finalizing] Applying {PERCENTILE_CUTOFF}th Percentile Filter & Exporting...")
    wt_clean = wt_um[wt_um <= np.percentile(wt_um, PERCENTILE_CUTOFF)]
    
    median_wt = float(np.median(wt_clean))
    median_d = float(np.median(skel_diams_um))
    wt_d_ratio = median_wt / median_d if median_d > 0 else 0

    # ── 6. Reporting ─────────────────────────────────────────────
    report = {
        "bifurcations": n_bif,
        "endpoints": n_end,
        "ep_bif_ratio": float(n_end/n_bif) if n_bif > 0 else 0,
        "median_diam_um": round(median_d, 2),
        "median_wt_um": round(median_wt, 2),
        "wt_d_ratio": round(wt_d_ratio, 4)
    }

    with open(str(REPORT_FILE), "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n── FINAL RESULTS ───────────────────────────────")
    print(f"Bifurcations : {n_bif:,}")
    print(f"Endpoints    : {n_end:,}")
    print(f"Median Diam  : {median_d:.1f} μm")
    print(f"Median WT    : {median_wt:.1f} μm")
    print(f"WT/D Ratio   : {wt_d_ratio:.3f}")
    print("═" * 60)
    print(f"✓ Total processing time: {(time.time() - t_start)/60:.1f} minutes.")

if __name__ == "__main__":
    run_biomarker_pipeline()
