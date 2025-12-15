import nrrd
import numpy as np
from scipy.ndimage import median_filter
import os

# === PARAMETERS ===
# input_path = '/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_reg_moments/US.nrrd'
input_path = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/trans1/US.nrrd'
output_path = os.path.join(os.path.dirname(input_path), 'US_surface.nrrd')

T = 20    # threshold on gradient magnitude
Th = 5    # thickness beyond the first hit

# === LOAD US VOLUME ===
us_data, us_header = nrrd.read(input_path)
us_data = us_data.astype(np.float32)

# === COMPUTE GRADIENT MAGNITUDE ===
gx, gy, gz = np.gradient(us_data)
grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

# === SURFACE EXTRACTION ===
mask = grad_mag > T
us_surf = np.zeros_like(us_data)

# Iterate slice by slice (axial: top -> bottom)
for k in range(grad_mag.shape[2]):
    slice_ = grad_mag[:, :, k]
    mask_slice = mask[:, :, k]

    for i in range(slice_.shape[0]):  # left-right
        v = mask_slice[i, ::-1]  # reverse top-to-bottom
        if not np.any(v):
            continue
        w = np.where(v)[0]
        cutoff = w[0] + Th
        cutoff = min(cutoff, slice_.shape[1])
        column_mask = np.zeros_like(v)
        column_mask[:cutoff] = 1
        column_mask = column_mask[::-1]  # flip mask back
        slice_[i, :] = slice_[i, :] * column_mask

    us_surf[:, :, k] = slice_ * mask_slice

# === APPLY 3D MEDIAN FILTER TO REMOVE OUTLIERS ===
us_surf_filtered = median_filter(us_surf, size=3)

# === SAVE AS NRRD ===
nrrd.write(output_path, us_surf_filtered.astype(np.float32), header=us_header)

print(f"Saved US surface volume to: {output_path}")
