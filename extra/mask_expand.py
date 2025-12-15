import nibabel as nib
import numpy as np
import os

# === PARAMETERS ===
input_nifti_path = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT.nii'
input_mask_path = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/bt_stitch.nii.gz'
output_nifti_path = os.path.join(os.path.dirname(input_nifti_path), 'CT_masked_th.nii.gz')

Th = 10  # number of slices to expand along the slice (z) axis

# === LOAD CT VOLUME AND MASK ===
ct_nii = nib.load(input_nifti_path)
ct = ct_nii.get_fdata()

mask_nii = nib.load(input_mask_path)
mask = mask_nii.get_fdata() > 0  # ensure binary

# === EXPAND MASK ALONG SLICE (Z) AXIS ===
expanded_mask = np.copy(mask)
for shift in range(1, Th + 1):
    expanded_mask[:, :, shift:] |= mask[:, :, :-shift]  # propagate mask forward along slices

# === APPLY MASK TO CT ===
ct_masked = ct * expanded_mask

# === SAVE RESULT ===
masked_nii = nib.Nifti1Image(ct_masked.astype(np.float32), ct_nii.affine, ct_nii.header)
nib.save(masked_nii, output_nifti_path)

print(f"Saved masked CT with expanded thickness {Th} slices to: {output_nifti_path}")
