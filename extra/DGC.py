import os
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import torch
import nrrd 
import SimpleITK as sitk


sigma = 1.5 
force_rigid = True 

# File paths
# CT_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/CT.nrrd"
# moving_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/co_registered/US.nrrd"
# moving_file = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/original_data/US_silvertruth.nrrd"
# moving_file = '/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/trans1/US.nrrd'
US_file = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US.nrrd'
transform = sitk.ReadTransform('/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/acq0_modif-ImageToReference.h5')
mha_US = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/percut/acq0_modif.igs.mha'
US_mask = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_mask.nrrd'

# Get the 3x3 rotation matrix
R = np.array(transform.GetMatrix()).reshape(3,3)
t = np.array(transform.GetTranslation())

x_us = np.array([1,0,0])  # probe’s X axis (lateral)
y_us = np.array([0,1,0])  # probe’s Y axis (elevational)
z_us = np.array([0,0,1])  # probe’s Z axis (along beam / depth)

x_ref = R @ x_us
y_ref = R @ y_us
z_ref = R @ z_us




# Load NRRD
# CT_sitk = sitk.ReadImage(CT_file)
US_sitk = sitk.ReadImage(US_file)
US_mask_sitk = sitk.ReadImage(US_mask)

dir_matrix = np.array(US_sitk.GetDirection()).reshape(3,3)
spacing = np.array(US_sitk.GetSpacing())
origin = np.array(US_sitk.GetOrigin())

# Physical => voxel coordinates
def phys_to_voxel(phys_vector):
    return np.linalg.inv(dir_matrix) @ (phys_vector / spacing)

x_voxel = phys_to_voxel(x_ref)
y_voxel = phys_to_voxel(y_ref)
z_voxel = phys_to_voxel(z_ref)

# CT_np = sitk.GetArrayFromImage(CT_sitk)  # Z x Y x X
US_np = sitk.GetArrayFromImage(US_sitk)
US_mask_np = sitk.GetArrayFromImage(US_mask_sitk)

# Reorder to X x Y x Z for NrrdParser
# CT_np = CT_np.transpose(2, 1, 0)
US_np = US_np.transpose(2, 1, 0)
US_mask_np = US_mask_np.transpose(2, 1, 0)

# Extract spacing
# CT_spacing = CT_sitk.GetSpacing()  # (dx, dy, dz)
US_spacing = US_sitk.GetSpacing()

# # gaussian smooth
# CT_smooth = smooth(CT_np, CT_spacing, sigma)
# US_smooth = smooth(US_np, US_spacing, sigma) # torch tensor

# # Compute gradient and Hessian
# grad_CT, _ = fd_3d_volume_derivatives(CT_smooth)
# grad_US, _ = fd_3d_volume_derivatives(US_smooth)


# grad_mag = torch.norm(grad_US, dim=-1)
# print(grad_mag.shape)

# Normalize the depth vector in voxel space (direction of the beam)
z_voxel_norm = z_voxel / np.linalg.norm(z_voxel)



# === Apply Depth Gain Compensation (DGC) using true beam direction inside mask ===

# Convert mask to torch tensor
US_mask_t = torch.tensor(US_mask_np > 0, dtype=torch.float32)  # 1 inside mask, 0 outside

# Compute depth map as before
nx, ny, nz = US_np.shape
xx, yy, zz = torch.meshgrid(
    torch.arange(nx, dtype=torch.float32),
    torch.arange(ny, dtype=torch.float32),
    torch.arange(nz, dtype=torch.float32),
    indexing="ij"
)
coords = torch.stack((xx, yy, zz), dim=-1)  # (nx, ny, nz, 3)

z_dir = torch.tensor(z_voxel_norm, dtype=torch.float32)

# compute depth map along probe's beam direction and normalize
depth_map = coords @ z_dir  # (nx, ny, nz)
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

# new sigmoid weighting
masked_depth = depth_map[US_mask_t > 0]
# Midpoint of depth within the mask
z0 = masked_depth.mean()  
# Then normalize depth relative to min/max inside mask if you want
depth_min_new = masked_depth.min()
depth_max_new = masked_depth.max()
depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

# k = 5.0 # controls slope
# w = 1 / (1 + torch.exp(-k * (depth_norm - z0)))
# # new gradient magnitude volume
# grad_mag_dgc = grad_mag * w

# 
z0 = masked_depth.mean()
k = 0.1  # smaller number because depth values are large
sigmoid_raw = 1 / (1 + torch.exp(-k * (depth_map - z0)))
sigmoid_at_mid = 1 / (1 + torch.exp(-k * (z0 - z0)))  # = 0.5
w = sigmoid_raw / sigmoid_at_mid
# grad_mag_dgc = grad_mag * w

# intensity weighted
US_torch = torch.tensor(US_np, dtype=torch.float32)
intensity_mag_dgc = US_torch * w

# VISUALLY INSPECT
x_idx = US_np.shape[0] // 2 - 20

# Extract the coronal slices
grad_slice = US_torch[:, x_idx, :].cpu().numpy()
grad_dgc_slice_newmask = intensity_mag_dgc[:, x_idx, :].cpu().numpy()

# Save as PNG images
plt.figure(figsize=(5,5))
plt.imshow(grad_slice.T, cmap='gray', origin='lower')
plt.title('Intensity (No DGC)')
plt.axis('off')
# plt.close()


plt.figure(figsize=(5,5))
plt.imshow(grad_dgc_slice_newmask.T, cmap='gray', origin='lower')
plt.title('Intensity (With New DGC)')
plt.axis('off')
# plt.close()
plt.show(block=True)


# out_dir = "/Users/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration"
# os.makedirs(out_dir, exist_ok=True)
# save_torch_as_nrrd(grad_mag_dgc, US_sitk, os.path.join(out_dir, "US_mag_weight.nrrd"))



# # COMPTUE PERCENTILE GRADIENTS
# value = 0.95 # percentile to retain
# threshold = torch.quantile(grad_mag_dgc, value) 
# # get mask using the DGC 
# percentile_mask = grad_mag_dgc >= threshold  
# combined_mask = percentile_mask
# combined_mask = combined_mask.unsqueeze(-1).expand_as(grad_US)

# # gradident thresholded
# grad_moving_percentile = grad_US.clone()
# grad_moving_percentile[~combined_mask] = 0.0

# # print(grad_US.shape)

# # magnitude thresholded
# grad_mag_thresh = grad_mag_dgc.clone()
# grad_mag_thresh[grad_mag_dgc < threshold] = 0.0

# # masked volume with same threshold
# US_tensor = torch.tensor(US_np, dtype=torch.float32)
# masked_us = US_tensor.clone()
# masked_us[~combined_mask[..., 0]] = 0.0  # combined_mask has shape X x Y x Z x 3



# SAVE OUTPUTS
out_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration"
os.makedirs(out_dir, exist_ok=True)

# save magnitude percentile thresholded
# save_torch_as_nrrd(grad_mag_thresh, US_sitk, os.path.join(out_dir, "US_mag_percentile.nrrd"))
# # save_torch_as_nrrd(masked_us, US_sitk, os.path.join(out_dir, "US_surface.nrrd"))

# save intensity percentile thresholded
# save_torch_as_nrrd(masked_us, US_sitk, os.path.join(out_dir, "US_surface.nrrd"))

# save intensity weighted volume
# save_torch_as_nrrd(intensity_mag_dgc, US_sitk, os.path.join(out_dir, "US_weight.nrrd"))

# # save gradient weighted volume
# save_torch_as_nrrd(grad_moving_percentile, US_sitk, os.path.join(out_dir, "US_grad.nrrd"))

