import SimpleITK as sitk
import numpy as np
import os

# === INPUTS ===
volume_path = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT.nrrd"
seg_path = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/bt_stitch.nrrd"
output_path = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_surface.nrrd"

# === LOAD IMAGES ===
volume = sitk.ReadImage(volume_path)
segmentation = sitk.ReadImage(seg_path)

# === CONVERT TO NUMPY ARRAYS ===
vol_arr = sitk.GetArrayFromImage(volume)        # shape: (Z, Y, X)
seg_arr = sitk.GetArrayFromImage(segmentation)  # shape: (Z, Y, X)

# Ensure segmentation is binary (1 inside, 0 outside)
mask = (seg_arr > 0).astype(np.uint8)

# === CHECK SHAPE COMPATIBILITY ===
if vol_arr.shape != mask.shape:
    raise ValueError(f"Shape mismatch: volume={vol_arr.shape}, mask={mask.shape}. "
                     "Make sure segmentation and CT are on the same grid in Slicer.")

# === APPLY MASK ===
masked_arr = np.where(mask == 1, vol_arr, 0)  # keep original intensities inside mask

# === CONVERT BACK TO SITK IMAGE ===
masked_img = sitk.GetImageFromArray(masked_arr)

# Copy spatial metadata (origin, spacing, direction) from the CT volume
masked_img.CopyInformation(volume)

# === SAVE MASKED VOLUME ===
sitk.WriteImage(masked_img, output_path)
print("Masked volume saved to:", output_path)
