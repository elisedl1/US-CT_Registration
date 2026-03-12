"""
Convert STL mesh files to binary NRRD label volumes,
using a reference CT NRRD for grid geometry.

Approach:
  1. Densely sample points on the mesh surface
  2. Map them to voxel indices via the NRRD's full affine (handles oblique grids)
  3. Fill the interior with scipy binary_fill_holes

Requirements:
    pip install numpy pynrrd trimesh scipy
"""

import numpy as np
import nrrd
import re
import trimesh
from scipy import ndimage
from pathlib import Path


def load_reference_grid(nrrd_path: str) -> dict:
    """Extract grid geometry from a reference NRRD volume."""
    header = nrrd.read_header(nrrd_path)
    data, _ = nrrd.read(nrrd_path)

    shape = np.array(data.shape)
    origin = np.array(header.get("space origin", [0.0, 0.0, 0.0]), dtype=np.float64)
    dirs = np.array(header["space directions"], dtype=np.float64)  # (3, 3)
    spacing = np.linalg.norm(dirs, axis=1)

    # World-to-voxel: ijk = (world - origin) @ inv(dirs)
    inv_dirs = np.linalg.inv(dirs)

    return {
        "shape": shape,
        "origin": origin,
        "directions": dirs,
        "inv_directions": inv_dirs,
        "spacing": spacing,
        "header": header,
    }


def stl_to_nrrd_volume(stl_path: str, ref: dict) -> np.ndarray:
    """
    Voxelise an STL mesh onto the reference grid.

    Densely samples the mesh surface, converts to voxel indices,
    then fills the interior.
    """
    mesh = trimesh.load_mesh(stl_path)
    print(f"  Mesh bounds:\n    min {mesh.bounds[0]}\n    max {mesh.bounds[1]}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    shape = ref["shape"]
    origin = ref["origin"]
    inv_dirs = ref["inv_directions"]
    min_spacing = ref["spacing"].min()

    # --- Dense surface sampling ---
    # Scale sample count to mesh surface area / voxel size
    surface_area = mesh.area
    samples_needed = int(surface_area / (min_spacing * 0.25) ** 2)
    samples_needed = max(samples_needed, 500_000)
    print(f"  Sampling {samples_needed:,} surface points...")

    surface_pts = mesh.sample(samples_needed)
    # Include vertices too
    all_pts = np.vstack([surface_pts, mesh.vertices])

    # --- World → voxel index ---
    ijk_float = (all_pts - origin) @ inv_dirs
    ijk_int = np.round(ijk_float).astype(np.int64)

    # Filter points within volume bounds
    valid = np.all((ijk_int >= 0) & (ijk_int < shape), axis=1)
    ijk_int = ijk_int[valid]
    print(f"  Points inside volume bounds: {len(ijk_int):,}")

    # --- Create surface mask ---
    vol = np.zeros(shape, dtype=np.uint8)
    vol[ijk_int[:, 0], ijk_int[:, 1], ijk_int[:, 2]] = 1
    surface_voxels = vol.sum()
    print(f"  Surface voxels: {surface_voxels:,}")

    # --- Fill interior ---
    if mesh.is_watertight:
        vol = ndimage.binary_fill_holes(vol).astype(np.uint8)
        print(f"  After fill: {vol.sum():,} voxels")
    else:
        print("  WARNING: mesh not watertight, skipping interior fill")

    return vol


def convert_stl_to_nrrd(stl_path: str, ref: dict, output_path: str | None = None):
    """Full pipeline: voxelise + write NRRD with matching header."""
    stl_path = str(stl_path)
    if output_path is None:
        output_path = str(Path(stl_path).with_suffix(".nrrd"))

    print(f"\nProcessing: {Path(stl_path).name}")
    vol = stl_to_nrrd_volume(stl_path, ref)

    out_header = {
        "space": ref["header"].get("space", "left-posterior-superior"),
        "space directions": ref["directions"].tolist(),
        "space origin": ref["origin"].tolist(),
        "encoding": "gzip",
    }
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
        match = re.search(r"(L\d+)", stl.name)
        if match:
            out_name = f"CT_{match.group(1)}.nrrd"
        else:
            out_name = stl.with_suffix(".nrrd").name
        out_path = str(DATA_DIR / out_name)
        convert_stl_to_nrrd(str(stl), ref, output_path=out_path)

    print("\nDone – all STL files converted.")