"""
Compare intensities at binary mask (CT segmentation) voxel locations
against two US images, after resampling everything into the same space.

Usage:
    python compare_mask_intensities.py
"""

import numpy as np
import SimpleITK as sitk

# ── Paths ─────────────────────────────────────────────────────────────────────
CT_MASK_PATH = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2"
    "/Registration/CT_segmentations/original/CT_seg_combined.nrrd"
)
US_PATHS = {
    "aniso_gabor": (
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2"
        "/Registration/US_Vertevra_axial_two_cal/aniso/aniso_gabor.nrrd"
    ),
    "US_complete_cal_preprocessed": (
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2"
        "/Registration/Known_Trans/sofa6/Cases/US_complete_cal_preprocessed.nrrd"
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> sitk.Image:
    img = sitk.ReadImage(path)
    print(f"  Loaded  : {path.split('/')[-1]}")
    print(f"    Size  : {img.GetSize()}")
    print(f"    Spacing: {img.GetSpacing()}")
    print(f"    Origin : {img.GetOrigin()}")
    return img


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample *moving* into the physical space defined by *reference*."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())  # identity – images share a coordinate frame
    return resampler.Execute(moving)


def compare(mask_arr: np.ndarray, image_arr: np.ndarray, label: str) -> None:
    """
    Extract the intensities of *image_arr* at every voxel where the
    binary mask is non-zero, then print summary statistics.
    """
    mask_bool = mask_arr > 0
    n_voxels = int(mask_bool.sum())

    if n_voxels == 0:
        print(f"\n[{label}]  Mask contains no foreground voxels – nothing to compare.")
        return

    intensities = image_arr[mask_bool].astype(float)

    print(f"\n{'='*60}")
    print(f"US image : {label}")
    print(f"{'='*60}")
    print(f"  Foreground voxels in mask : {n_voxels:,}")
    print(f"  Intensity  min            : {intensities.min():.4f}")
    print(f"  Intensity  max            : {intensities.max():.4f}")
    print(f"  Intensity  mean           : {intensities.mean():.4f}")
    print(f"  Intensity  median         : {np.median(intensities):.4f}")
    print(f"  Intensity  std            : {intensities.std():.4f}")
    print(f"  Intensity  25th pct       : {np.percentile(intensities, 25):.4f}")
    print(f"  Intensity  75th pct       : {np.percentile(intensities, 75):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Loading images ──────────────────────────────────────────────")
    ct_mask = load(CT_MASK_PATH)

    # Cast mask to uint8 so it's unambiguously binary
    ct_mask = sitk.Cast(ct_mask, sitk.sitkUInt8)

    # Use the CT mask as the reference space; everything will be resampled into it.
    reference = ct_mask
    mask_arr = sitk.GetArrayFromImage(ct_mask)  # shape: (z, y, x)

    print(f"\n  Mask unique values: {np.unique(mask_arr)}")
    print(f"  Mask foreground voxels: {(mask_arr > 0).sum():,}")

    for name, us_path in US_PATHS.items():
        print(f"\n── Processing {name} ──────────────────────────────────────────")
        us_img = load(us_path)

        # Resample US into CT-mask space (nearest-neighbour for speed is fine for
        # intensity look-up; linear gives smoother sub-voxel estimates)
        print(f"  Resampling into CT mask space …")
        us_resampled = resample_to_reference(us_img, reference, interpolator=sitk.sitkLinear)

        us_arr = sitk.GetArrayFromImage(us_resampled).astype(float)

        compare(mask_arr, us_arr, label=name)

    print(f"\n{'='*60}\nDone.\n")


if __name__ == "__main__":
    main()