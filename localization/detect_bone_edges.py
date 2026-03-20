"""
detect_bone_edges.py

Bone edge detection pipeline for preprocessed 3-D ultrasound (NRRD).

Pipeline per axial frame (Z-slice):
  1. Sobel edge detection
  2. Binary threshold on edge magnitude
  3. Island removal   (remove small connected components)
  4. Erosion
  5. Dilation

Output
------
  <stem>_bone_candidates.nrrd   – uint8 binary mask of candidate bone edges

Usage
-----
  python detect_bone_edges.py
  python detect_bone_edges.py --input vol.nrrd --mask mask.nrrd
"""

import argparse
import numpy as np
import nrrd
from scipy.ndimage import (
    sobel,
    label,
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
)

# ──────────────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────────────
_BASE = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/"
    "Registration/US_Vertebra_axial_cal/"
)
DEFAULT_INPUT    = _BASE + "US_complete_cal_preprocessed.nrrd"
DEFAULT_MASK     = _BASE + "US_complete_cal_mask.nrrd"
DEFAULT_BONE_OUT = _BASE + "US_complete_cal_bone_candidates.nrrd"

# ──────────────────────────────────────────────────────────────────────────────
# Tunable parameters
# ──────────────────────────────────────────────────────────────────────────────
PARAMS = dict(
    edge_threshold     = 0.1,  # fraction of per-slice max edge magnitude
    min_island_area    = 60,    # px; blobs smaller than this are dropped
    erosion_iterations = 2,
    dilation_iterations= 2,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sobel_magnitude_2d(img: np.ndarray) -> np.ndarray:
    sx  = sobel(img.astype(np.float64), axis=0)
    sy  = sobel(img.astype(np.float64), axis=1)
    mag = np.hypot(sx, sy)
    m   = mag.max()
    return (mag / m) if m > 0 else mag


def remove_small_islands(binary: np.ndarray, min_area: int) -> np.ndarray:
    struct          = generate_binary_structure(2, 2)   # 8-connectivity
    labeled, _      = label(binary, structure=struct)
    sizes           = np.bincount(labeled.ravel())
    keep            = np.where(sizes >= min_area)[0]
    keep            = keep[keep != 0]
    return np.isin(labeled, keep)


# ──────────────────────────────────────────────────────────────────────────────
# Per-slice pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_slice(
    img:  np.ndarray,   # 2-D float32, shape (rows, cols)
    mask: np.ndarray,   # 2-D uint8
    p:    dict,
) -> np.ndarray:
    """Returns uint8 bone-candidate mask for one slice."""
    if mask.sum() == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    # 1. Sobel edge magnitude
    edges = sobel_magnitude_2d(img)

    # 2. Binary threshold — restrict to image mask
    emax     = edges.max()
    edge_bin = (edges > p["edge_threshold"] * emax) if emax > 0 else np.zeros_like(edges, bool)
    edge_bin = edge_bin & (mask > 0)

    # 3. Island removal
    edge_bin = remove_small_islands(edge_bin, p["min_island_area"])

    # 4. Erosion
    struct   = generate_binary_structure(2, 1)   # 4-connectivity structuring element
    edge_bin = binary_erosion(edge_bin,  structure=struct, iterations=p["erosion_iterations"])

    # 5. Dilation
    edge_bin = binary_dilation(edge_bin, structure=struct, iterations=p["dilation_iterations"])

    return edge_bin.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(input_path, mask_path, bone_out, p=PARAMS):
    print(f"Loading volume : {input_path}")
    volume, header = nrrd.read(input_path)
    print(f"  Shape : {volume.shape}   dtype : {volume.dtype}")

    print(f"Loading mask   : {mask_path}")
    mask_vol, _ = nrrd.read(mask_path)

    # Normalise to [0, 1]
    vmin, vmax = float(volume.min()), float(volume.max())
    vol_f = (volume.astype(np.float32) - vmin) / (vmax - vmin + 1e-9)

    n_z      = vol_f.shape[2]
    bone_vol = np.zeros(vol_f.shape, dtype=np.uint8)

    print(f"Processing {n_z} axial slices …")
    for z in range(n_z):
        if z % 20 == 0:
            print(f"  {z}/{n_z}", end="\r", flush=True)
        bone_vol[:, :, z] = process_slice(vol_f[:, :, z], mask_vol[:, :, z], p)

    print(f"\nCandidate bone-edge voxels: {int(bone_vol.sum())}")

    bone_hdr = header.copy()
    bone_hdr["type"] = "unsigned char"
    bone_hdr.setdefault("encoding", "gzip")

    print(f"Saving : {bone_out}")
    nrrd.write(bone_out, bone_vol, bone_hdr)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bone edge detection pipeline for preprocessed US NRRD volume."
    )
    parser.add_argument("--input",    "-i", default=DEFAULT_INPUT)
    parser.add_argument("--mask",     "-m", default=DEFAULT_MASK)
    parser.add_argument("--bone-out", "-b", default=DEFAULT_BONE_OUT)
    for k, v in PARAMS.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)

    args   = parser.parse_args()
    p      = {k: getattr(args, k) for k in PARAMS}
    run(args.input, args.mask, args.bone_out, p)