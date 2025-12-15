import nibabel as nib
import numpy as np
import os

# === PARAMETERS ===
input_nifti_path = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT.nii'
input_mask_path = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/bt_stitch.nii.gz'
output_nifti_path = os.path.join(os.path.dirname(input_nifti_path), 'CT_masked_th.nii.gz')

Th = 3  # number of slices to expand in the slice direction

# === LOAD CT VOLUME AND MASK ===
ct_nii = nib.load(input_nifti_path)
ct = ct_nii.get_fdata()

mask_nii = nib.load(input_mask_path)
mask = mask_nii.get_fdata() > 0  # ensure binary

# === EXPAND MASK ALONG SLICE DIRECTION ===
ct_masked = np.zeros_like(ct)

for k in range(ct.shape[2]):
    # Determine the slices to include
    start = k
    end = min(k + Th, ct.shape[2])
    
    # If any voxel in the current slice is in the mask, include next Th slices
    if np.any(mask[:, :, k]):
        ct_masked[:, :, start:end] = ct[:, :, start:end]

# Apply original mask to avoid including voxels outside mask in the first slice
ct_masked *= mask

# === SAVE RESULT ===
masked_nii = nib.Nifti1Image(ct_masked.astype(np.float32), ct_nii.affine, ct_nii.header)
nib.save(masked_nii, output_nifti_path)

print(f"Saved masked CT with thickness {Th} to: {output_nifti_path}")
