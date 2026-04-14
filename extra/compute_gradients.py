"""
Compute 3D image gradients for US and CT posterior surface volumes.
Outputs:
  - US gradient magnitude volume
  - US gradient vectors (x, y, z components)
  - US gradient magnitude filtered to 80th percentile (binary mask + magnitudes)
  - CT gradient magnitude volume (per vertebra)
  - CT gradient vectors (per vertebra)
"""

import numpy as np
import nrrd
from scipy.ndimage import gaussian_filter
import os
import argparse


# ── paths ──────────────────────────────────────────────────────────────────
US_PATH = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/aniso/aniso_preprocessed_gabor.nrrd"

CT_PATHS = {
    "L1": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa5/Cases/all_moving/L1_post.nrrd",
    "L2": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa5/Cases/all_moving/L2_post.nrrd",
    "L3": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa5/Cases/all_moving/L3_post.nrrd",
    "L4": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa5/Cases/all_moving/L4_post.nrrd",
}

OUT_DIR = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/gradients/sofa5_CT"

# ── parameters ─────────────────────────────────────────────────────────────
SIGMA = 1.0          # Gaussian smoothing sigma before derivative (in voxels)
PERCENTILE = 95      # gradient magnitude percentile threshold for US


def get_spacing_from_header(header):
    """Extract voxel spacing (mm) from NRRD header."""
    if "spacings" in header:
        return np.array(header["spacings"])
    if "space directions" in header:
        dirs = header["space directions"]
        # each row is a direction vector; spacing is its norm
        spacing = np.array([np.linalg.norm(d) for d in dirs])
        return spacing
    # fallback: isotropic 1mm
    return np.ones(3)


def compute_gradients(volume, spacing, sigma=1.0):
    """
    Compute 3D image gradients using Gaussian derivative.
    
    Args:
        volume:  3D numpy array (float)
        spacing: voxel size in mm, shape (3,) — order matches volume axes
        sigma:   Gaussian smoothing sigma in voxels
    
    Returns:
        grad_x, grad_y, grad_z : gradient components (same shape as volume)
        magnitude              : gradient magnitude
    """
    vol = volume.astype(np.float32)

    # Gaussian derivative: smooth then finite-difference along each axis
    # scipy gaussian_filter1d with order=1 gives the Gaussian derivative
    from scipy.ndimage import gaussian_filter1d

    grad_x = gaussian_filter1d(vol, sigma=sigma, axis=0, order=1) / spacing[0]
    grad_y = gaussian_filter1d(vol, sigma=sigma, axis=1, order=1) / spacing[1]
    grad_z = gaussian_filter1d(vol, sigma=sigma, axis=2, order=1) / spacing[2]

    magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return grad_x, grad_y, grad_z, magnitude


def percentile_mask(magnitude, percentile=80):
    """Return boolean mask of voxels above the given percentile of magnitude."""
    threshold = np.percentile(magnitude, percentile)
    print(f"  Gradient magnitude {percentile}th percentile threshold: {threshold:.4f}")
    return magnitude >= threshold, threshold


def save_nrrd(array, header_ref, path, description=""):
    """Save a numpy array as NRRD, copying spatial metadata from a reference header."""
    out_header = {}

    for key in ["space", "space directions", "space origin", "spacings"]:
        if key in header_ref:
            out_header[key] = header_ref[key]

    out_header["type"] = "float"
    out_header["dimension"] = array.ndim
    out_header["sizes"] = list(array.shape)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    nrrd.write(path, array.astype(np.float32), out_header)
    print(f"  Saved {description}: {path}")


def save_gradient_vector_nrrd(gx, gy, gz, header_ref, path, description="gradient vector field"):
    """
    Save (gx, gy, gz) as a single 4D NRRD with shape (3, X, Y, Z).
    The 'kinds' field marks axis 0 as a 3-vector so Slicer/ITK-SNAP
    interpret it correctly as a vector volume.
    """
    # stack along a new leading axis → (3, X, Y, Z)
    vec = np.stack([gx, gy, gz], axis=0).astype(np.float32)

    out_header = {}
    out_header["type"] = "float"
    out_header["dimension"] = 4
    out_header["sizes"] = list(vec.shape)

    # kinds: first axis is the vector components, rest are spatial
    out_header["kinds"] = ["vector", "space", "space", "space"]

    # space directions: NaN row for the vector axis, then copy spatial directions
    # pynrrd requires a homogeneous numpy array, so we use a NaN row for the non-spatial axis
    if "space directions" in header_ref:
        spatial_dirs = np.array(header_ref["space directions"], dtype=float)  # (3, 3)
        nan_row = np.full((1, spatial_dirs.shape[1]), np.nan)                 # (1, 3)
        out_header["space directions"] = np.concatenate([nan_row, spatial_dirs], axis=0)  # (4, 3)
    if "space" in header_ref:
        out_header["space"] = header_ref["space"]
    if "space origin" in header_ref:
        out_header["space origin"] = header_ref["space origin"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    nrrd.write(path, vec, out_header)
    print(f"  Saved {description}: {path}  shape={vec.shape}")


def process_volume(volume, header, spacing, sigma, label, out_subdir, percentile=None):
    """
    Compute and save gradients for a single volume.
    
    If percentile is given, also saves the magnitude masked to that percentile.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"  Volume shape : {volume.shape}")
    print(f"  Spacing (mm) : {spacing}")
    print(f"  Intensity range: [{volume.min():.2f}, {volume.max():.2f}]")

    gx, gy, gz, mag = compute_gradients(volume, spacing, sigma=sigma)

    print(f"  Gradient magnitude range: [{mag.min():.4f}, {mag.max():.4f}]")

    base = os.path.join(out_subdir, label)
    os.makedirs(base, exist_ok=True)

    save_nrrd(mag, header, os.path.join(base, "gradient_magnitude.nrrd"), "gradient magnitude")
    save_gradient_vector_nrrd(gx, gy, gz, header,
                              os.path.join(base, "gradient_vectors.nrrd"),
                              "gradient vector field (3, X, Y, Z)")

    if percentile is not None:
        mask, thresh = percentile_mask(mag, percentile)
        # save as float: 0 or magnitude value (so you can see both mask and strength)
        mag_filtered = np.where(mask, mag, 0.0)
        mask_float   = mask.astype(np.float32)

        save_nrrd(mag_filtered, header,
                  os.path.join(base, f"gradient_magnitude_p{percentile}.nrrd"),
                  f"gradient magnitude >{percentile}th pctile")
        save_nrrd(mask_float,   header,
                  os.path.join(base, f"gradient_mask_p{percentile}.nrrd"),
                  f"gradient binary mask >{percentile}th pctile")

        # masked vector field: zero out vectors below the percentile threshold
        gx_masked = np.where(mask, gx, 0.0)
        gy_masked = np.where(mask, gy, 0.0)
        gz_masked = np.where(mask, gz, 0.0)
        save_gradient_vector_nrrd(gx_masked, gy_masked, gz_masked, header,
                                  os.path.join(base, f"gradient_vectors_p{percentile}.nrrd"),
                                  f"gradient vector field >{percentile}th pctile (3, X, Y, Z)")

        n_selected = int(mask.sum())
        n_total    = int(mask.size)
        print(f"  Selected {n_selected}/{n_total} voxels ({100*n_selected/n_total:.1f}%)")

    return gx, gy, gz, mag


# ── main ───────────────────────────────────────────────────────────────────
def main():

    # # ---- US volume --------------------------------------------------------
    # print("\nLoading US volume...")
    # us_vol, us_header = nrrd.read(US_PATH)
    # us_spacing = get_spacing_from_header(us_header)

    # process_volume(
    #     volume=us_vol,
    #     header=us_header,
    #     spacing=us_spacing,
    #     sigma=SIGMA,
    #     label="US",
    #     out_subdir=OUT_DIR,
    #     percentile=PERCENTILE,
    # )

    # ---- CT posterior surface volumes ------------------------------------
    for vertebra, ct_path in CT_PATHS.items():
        print(f"\nLoading CT {vertebra} posterior surface...")
        ct_vol, ct_header = nrrd.read(ct_path)
        ct_spacing = get_spacing_from_header(ct_header)

        process_volume(
            volume=ct_vol,
            header=ct_header,
            spacing=ct_spacing,
            sigma=SIGMA,
            label=f"CT_{vertebra}",
            out_subdir=OUT_DIR,
            percentile=None,   # paper samples CT gradients at CT surface points, not percentile
        )

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {OUT_DIR}")
    print("\nOutputs per volume:")
    print("  gradient_magnitude.nrrd          : scalar magnitude, shape (X, Y, Z)")
    print("  gradient_vectors.nrrd            : vector field, shape (3, X, Y, Z) — use this for the dot product metric")
    print(f"  gradient_magnitude_p{PERCENTILE}.nrrd   : magnitude at strong voxels only (US only)")
    print(f"  gradient_mask_p{PERCENTILE}.nrrd         : binary mask of strong voxels (US only)")
    print("\nTo load gradient_vectors.nrrd in Python later:")
    print("  vec, _ = nrrd.read('gradient_vectors.nrrd')  # shape (3, X, Y, Z)")
    print("  gx, gy, gz = vec[0], vec[1], vec[2]")


if __name__ == "__main__":
    main()