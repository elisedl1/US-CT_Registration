import nibabel as nib
import numpy as np
import os

# === PARAMETERS ===
input_nifti_path = '/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/original_data/CT.nii'
output_nifti_path = os.path.join(os.path.dirname(input_nifti_path), 'CT_surface.nii.gz')

T = 270   # intensity threshold for bone
Th = 10   # thickness beyond the surface

# === LOAD CT VOLUME ===
nii = nib.load(input_nifti_path)
ct = nii.get_fdata()

# === BONE SURFACE EXTRACTION ===
mask = ct > T
ct_surf = np.zeros_like(ct)

# Iterate slice by slice (axial: inferior -> superior)
for k in range(ct.shape[2]):
    slice_ = ct[:, :, k]
    mask_slice = mask[:, :, k]

    # Process each column (scan top-to-bottom = anterior to posterior)
    for i in range(slice_.shape[0]):  # left-right
        v = mask_slice[i, ::-1]      # reverse anterior-posterior to go top-to-bottom
        if not np.any(v):
            continue
        w = np.where(v)[0]
        cutoff = w[0] + Th
        cutoff = min(cutoff, slice_.shape[1])
        column_mask = np.zeros_like(v)
        column_mask[:cutoff] = 1
        column_mask = column_mask[::-1]  # flip mask back
        slice_[i, :] = slice_[i, :] * column_mask

    ct_surf[:, :, k] = slice_ * mask_slice

# === SAVE RESULT ===
surf_nii = nib.Nifti1Image(ct_surf.astype(np.float32), nii.affine, nii.header)
nib.save(surf_nii, output_nifti_path)

print(f"Saved bone surface CT to: {output_nifti_path}")
