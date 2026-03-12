'''
Steps:
1) convert stl segmentations to nrrd volumes using nrrd CT volume
** allows me to then use the original extra/posterior_surface.py to get the binary masks
-> these are used as input in the actual registration algorithm

'''

"""
Convert STL mesh files to binary NRRD label volumes,
using a reference CT NRRD for grid geometry (size, spacing, origin, direction).

Requirements:
    pip install numpy nrrd trimesh
"""

import numpy as np
import nrrd
import re
import trimesh
from pathlib import Path


def load_reference_grid(nrrd_path: str) -> dict:
    """Extract grid geometry from a reference NRRD volume."""
    header = nrrd.read_header(nrrd_path)
    data, _ = nrrd.read(nrrd_path)

    shape = np.array(data.shape)  # (i, j, k)
    origin = np.array(header.get("space origin", [0.0, 0.0, 0.0]), dtype=np.float64)

    # Build voxel-to-world matrix from space directions (3x3)
    dirs = np.array(header["space directions"], dtype=np.float64)  # (3, 3)
    spacing = np.linalg.norm(dirs, axis=1)

    return {
        "shape": shape,
        "origin": origin,
        "directions": dirs,
        "spacing": spacing,
        "header": header,
    }


def stl_to_nrrd_volume(stl_path: str, ref: dict) -> np.ndarray:
    """
    Voxelise an STL mesh onto the reference grid.

    For every voxel centre we test whether it lies inside the mesh.
    Returns a uint8 volume (1 = inside, 0 = outside).
    """
    mesh = trimesh.load_mesh(stl_path)
    print(f"  Mesh bounds:\n    min {mesh.bounds[0]}\n    max {mesh.bounds[1]}")
    print(f"  Mesh is watertight: {mesh.is_watertight}")

    shape = ref["shape"]
    origin = ref["origin"]
    dirs = ref["directions"]  # columns = axis vectors (with spacing baked in)

    # Build every voxel-centre coordinate in world space
    # world_coord = origin + i*dir[0] + j*dir[1] + k*dir[2]
    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    # Flatten for batch query
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1).astype(np.float64)
    world_pts = origin + ijk @ dirs  # (N, 3)

    # ---- Crop to mesh bounding box first (huge speed-up) ----
    pad = ref["spacing"].max() * 2  # small safety margin
    bb_min = mesh.bounds[0] - pad
    bb_max = mesh.bounds[1] + pad
    inside_bb = np.all((world_pts >= bb_min) & (world_pts <= bb_max), axis=1)
    candidate_idx = np.where(inside_bb)[0]
    print(f"  Voxels inside bounding box: {len(candidate_idx)} / {len(world_pts)}")

    volume = np.zeros(len(world_pts), dtype=np.uint8)

    if len(candidate_idx) > 0:
        contains = mesh.contains(world_pts[candidate_idx])
        volume[candidate_idx] = contains.astype(np.uint8)

    print(f"  Voxels inside mesh: {volume.sum()}")
    return volume.reshape(shape)


def convert_stl_to_nrrd(stl_path: str, ref: dict, output_path: str | None = None):
    """Full pipeline: voxelise + write NRRD with matching header."""
    stl_path = str(stl_path)
    if output_path is None:
        output_path = str(Path(stl_path).with_suffix(".nrrd"))

    print(f"\nProcessing: {Path(stl_path).name}")
    vol = stl_to_nrrd_volume(stl_path, ref)

    # Build a minimal NRRD header that preserves the spatial metadata
    out_header = {
        "space": ref["header"].get("space", "left-posterior-superior"),
        "space directions": ref["directions"].tolist(),
        "space origin": ref["origin"].tolist(),
        "encoding": "gzip",
    }
    # Preserve optional fields if present
    for key in ("kinds", "measurement frame"):
        if key in ref["header"]:
            out_header[key] = ref["header"][key]

    nrrd.write(output_path, vol, header=out_header)
    print(f"  Saved → {output_path}")


# --------------- main ---------------
if __name__ == "__main__":
    DATA_DIR = Path(
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/"
        "cava_data/processed/Case1/CT"
    )
    CT_NRRD = DATA_DIR / "CT_cor.nrrd"

    # Collect all STL files in the directory
    STL_FILES = sorted(DATA_DIR.glob("*.stl"))

    if not STL_FILES:
        print(f"No .stl files found in {DATA_DIR}")
        exit(1)

    print("Loading reference CT grid...")
    ref = load_reference_grid(str(CT_NRRD))
    print(f"  Shape:   {ref['shape']}")
    print(f"  Spacing: {ref['spacing']}")
    print(f"  Origin:  {ref['origin']}")
    print(f"\nFound {len(STL_FILES)} STL file(s):")
    for p in STL_FILES:
        print(f"  {p.name}")

    for stl in STL_FILES:
        # Extract "LX" from names like "URS1_L1 11_001.stl"
        match = re.search(r"(L\d+)", stl.name)
        if match:
            out_name = f"CT_{match.group(1)}.nrrd"
        else:
            out_name = stl.with_suffix(".nrrd").name
        out_path = str(DATA_DIR / out_name)
        convert_stl_to_nrrd(str(stl), ref, output_path=out_path)

    print("\nDone – all STL files converted.")

"""
call:
python seg2nrrd.py

"""