import os
from functools import reduce

import numpy as np
import torch
from scipy.optimize import differential_evolution
import nrrd  # pynrrd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

from utils.finite_differences import (
    fd_3d_neighbourhood_derivatives,
    fd_3d_volume_derivatives,
    neighbourhood_indices
)

from utils.helpers import (
    compute_determinant,
    create_image_mask,
    normalize_hessian,
    transform_affine_3d,
    export_samples_to_slicer_json,
    monogenic_filter_3d,
    mean_curvature_3d,
    hessian_determinant,
    smooth,
    save_torch_as_nrrd,
    create_boundary_exclusion_mask
)

sigma = 1.5 
force_rigid = True 

# File paths
fixed_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/CT.nrrd"
# moving_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/US.nrrd"
# moving_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/original_data/US_silvertruth.nrrd"
# moving_file = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/trans1/US.nrrd'
moving_file = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US.nrrd'


# Load NRRD
fixed_sitk = sitk.ReadImage(fixed_file)
moving_sitk = sitk.ReadImage(moving_file)

fixed_np = sitk.GetArrayFromImage(fixed_sitk)  # Z x Y x X
moving_np = sitk.GetArrayFromImage(moving_sitk)

# Reorder to X x Y x Z for NrrdParser
fixed_np = fixed_np.transpose(2, 1, 0)
moving_np = moving_np.transpose(2, 1, 0)

# Extract spacing
fixed_spacing = fixed_sitk.GetSpacing()  # (dx, dy, dz)
moving_spacing = moving_sitk.GetSpacing()

# gaussian smooth
fixed_smooth = smooth(fixed_np, fixed_spacing, sigma)
moving_smooth = smooth(moving_np, moving_spacing, sigma) # torch tensor

# Compute gradient and Hessian
grad_fixed, hess_fixed = fd_3d_volume_derivatives(fixed_smooth)
grad_moving, hess_moving = fd_3d_volume_derivatives(moving_smooth)

# get 80th percentile gradients from US image
epsilon = 1e-5
bg = moving_np <= epsilon
erosion_size = 8
bg_dilated = binary_dilation(bg, iterations=erosion_size)
us_mask = ~bg_dilated  # invert to get interior mask
us_mask = torch.tensor(us_mask, dtype=torch.bool)

grad_mag = torch.norm(grad_moving, dim=-1)
threshold = torch.quantile(grad_mag, 0.95) # 95TH PERENTILE HARDCODED

percentile_mask = grad_mag >= threshold  
combined_mask = percentile_mask & us_mask
combined_mask = combined_mask.unsqueeze(-1).expand_as(grad_moving)
grad_moving_80 = grad_moving.clone()
grad_moving_80[~combined_mask] = 0.0

print(grad_moving.shape)

# magnitude thresholded
grad_mag_thresh = grad_mag.clone()
grad_mag_thresh[grad_mag < threshold] = 0.0

# masked volume with same thrtehsold
moving_tensor = torch.tensor(moving_np, dtype=torch.float32)
masked_us = moving_tensor.clone()
masked_us[~combined_mask[..., 0]] = 0.0  # combined_mask has shape X x Y x Z x 3

# out_dir = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/trans1"
out_dir = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/trans1"
os.makedirs(out_dir, exist_ok=True)
# save_torch_as_nrrd(grad_moving, moving_sitk, os.path.join(out_dir, "grad_moving_silvertruth.nrrd"))
save_torch_as_nrrd(grad_fixed, fixed_sitk, os.path.join(out_dir, "grad_fixed.nrrd"))

# save_torch_as_nrrd(grad_moving_80, moving_sitk, os.path.join(out_dir, "US_grads_80.nrrd"))
# save_torch_as_nrrd(grad_moving_80, moving_sitk, os.path.join(out_dir, "US_grads_80_silvertruth.nrrd"))
# save_torch_as_nrrd(grad_mag_thresh, moving_sitk, os.path.join(out_dir, "US_mag_80.nrrd"))

# save_torch_as_nrrd(grad_mag_thresh.unsqueeze(-1), moving_sitk, os.path.join(out_dir, "US_mag.nrrd"))


# save_torch_as_nrrd(masked_us, moving_sitk, os.path.join(out_dir, "US_surface.nrrd"))



# us_mask_float = us_mask.unsqueeze(-1).float()
# save_torch_as_nrrd(
#     us_mask_float,
#     moving_sitk,
#     os.path.join(out_dir, "US_boundary_mask.nrrd")
# )

