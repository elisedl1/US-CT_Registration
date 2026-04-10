"""
aniso_filter.py
---------------
Apply anisotropic diffusion and Gaussian smoothing filters to a 3D ultrasound
volume (.nrrd), print statistics, and save both outputs to the same directory
as the input file.

Usage:
    python aniso_filter.py [input_path]

Default input: /Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/
               Registration/US_Vertevra_axial_two_cal/US_complete_cal.nrrd
"""

import sys
import os
import time
import numpy as np
import SimpleITK as sitk


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_INPUT = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/log_gabor_enhanced.nrrd"
)

# Anisotropic diffusion (CurvatureAnisotropicDiffusion) parameters
ANISO_TIMESTEP        = 0.0625   # stable for 3-D when conductance is moderate
ANISO_CONDUCTANCE     = 3.0      # edge sensitivity (lower = sharper edges kept)
ANISO_ITERATIONS      = 10       # number of diffusion steps

# Gaussian smoothing parameters
GAUSS_SIGMA           = 1.5      # physical-space sigma (mm); adjust to taste


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_image(path: str) -> sitk.Image:
    print(f"\n{'='*60}")
    print(f"  Loading: {path}")
    print(f"{'='*60}")
    img = sitk.ReadImage(path)
    return img


def print_stats(label: str, img: sitk.Image) -> None:
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    spacing = img.GetSpacing()
    size    = img.GetSize()

    print(f"\n── {label} ──")
    print(f"  Size (x,y,z)    : {size}")
    print(f"  Spacing (mm)    : {tuple(f'{s:.4f}' for s in spacing)}")
    print(f"  Pixel type      : {img.GetPixelIDTypeAsString()}")
    print(f"  Min             : {arr.min():.4f}")
    print(f"  Max             : {arr.max():.4f}")
    print(f"  Mean            : {arr.mean():.4f}")
    print(f"  Std dev         : {arr.std():.4f}")
    print(f"  Median          : {float(np.median(arr)):.4f}")
    print(f"  25th pct        : {float(np.percentile(arr, 25)):.4f}")
    print(f"  75th pct        : {float(np.percentile(arr, 75)):.4f}")
    print(f"  Non-zero voxels : {int(np.count_nonzero(arr))}")


def cast_to_float(img: sitk.Image) -> sitk.Image:
    """Filters require float32 input."""
    return sitk.Cast(img, sitk.sitkFloat32)


def apply_anisotropic(img: sitk.Image) -> sitk.Image:
    print(f"\n{'='*60}")
    print("  Applying Anisotropic Diffusion filter …")
    print(f"    iterations  = {ANISO_ITERATIONS}")
    print(f"    time step   = {ANISO_TIMESTEP}")
    print(f"    conductance = {ANISO_CONDUCTANCE}")
    print(f"{'='*60}")

    flt = sitk.CurvatureAnisotropicDiffusionImageFilter()
    flt.SetNumberOfIterations(ANISO_ITERATIONS)
    flt.SetTimeStep(ANISO_TIMESTEP)
    flt.SetConductanceParameter(ANISO_CONDUCTANCE)

    t0 = time.time()
    result = flt.Execute(cast_to_float(img))
    print(f"  Done in {time.time()-t0:.1f} s")
    return result


def apply_gaussian(img: sitk.Image) -> sitk.Image:
    print(f"\n{'='*60}")
    print("  Applying Gaussian Smoothing filter …")
    print(f"    sigma = {GAUSS_SIGMA} mm (physical space)")
    print(f"{'='*60}")

    flt = sitk.SmoothingRecursiveGaussianImageFilter()
    flt.SetSigma(GAUSS_SIGMA)

    t0 = time.time()
    result = flt.Execute(cast_to_float(img))
    print(f"  Done in {time.time()-t0:.1f} s")
    return result


def save_image(img: sitk.Image, path: str) -> None:
    # Cast back to the same pixel type as input if desired, or keep float32
    sitk.WriteImage(img, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved → {path}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    if not os.path.isfile(input_path):
        sys.exit(f"ERROR: File not found: {input_path}")

    out_dir   = os.path.dirname(input_path)
    base_stem = os.path.splitext(os.path.basename(input_path))[0]
    ext       = ".nrrd"

    aniso_out = os.path.join(out_dir, f"{base_stem}_anisotropic{ext}")
    gauss_out = os.path.join(out_dir, f"{base_stem}_gaussian{ext}")

    # ── Load ──────────────────────────────────
    original = load_image(input_path)
    print_stats("Original", original)

    # # ── Anisotropic diffusion ─────────────────
    # aniso = apply_anisotropic(original)
    # # print_stats("After Anisotropic Diffusion", aniso)
    # save_image(aniso, aniso_out)

    # ── Gaussian smoothing ────────────────────
    gauss = apply_gaussian(original)
    # print_stats("After Gaussian Smoothing", gauss)
    save_image(gauss, gauss_out)

    # # ── Summary ───────────────────────────────
    # print(f"\n{'='*60}")
    # print("  Output files")
    # print(f"{'='*60}")
    # print(f"  Anisotropic : {aniso_out}")
    # print(f"  Gaussian    : {gauss_out}")
    # print(f"  Directory   : {out_dir}")
    # print()


if __name__ == "__main__":
    main()