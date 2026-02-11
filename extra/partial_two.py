import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk


# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose which side to blank out: 'right' or 'left'
BLANK_SIDE = 'left'

# File paths
US_FILE = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3/US_weight_L3.nrrd'
TRANSFORM_FILE = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/acq0_modif-ImageToReference.h5'
US_MASK_FILE = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3/US_weight_L3.nrrd'
OUT_DIR = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3"


# ============================================================================
# LOAD DATA
# ============================================================================

# Load transform
transform = sitk.ReadTransform(TRANSFORM_FILE)

# Get the 3x3 rotation matrix and translation
R = np.array(transform.GetMatrix()).reshape(3, 3)
t = np.array(transform.GetTranslation())

# Define probe's coordinate axes
x_us = np.array([1, 0, 0])  # probe's X axis (lateral)
y_us = np.array([0, 1, 0])  # probe's Y axis (elevational)
z_us = np.array([0, 0, 1])  # probe's Z axis (along beam / depth)

# Transform to reference frame
x_ref = R @ x_us
y_ref = R @ y_us
z_ref = R @ z_us

# Load images
US_sitk = sitk.ReadImage(US_FILE)
US_mask_sitk = sitk.ReadImage(US_MASK_FILE)

# Get image properties
dir_matrix = np.array(US_sitk.GetDirection()).reshape(3, 3)
spacing = np.array(US_sitk.GetSpacing())
origin = np.array(US_sitk.GetOrigin())


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def phys_to_voxel(phys_vector):
    """Convert physical coordinates to voxel coordinates."""
    return np.linalg.inv(dir_matrix) @ (phys_vector / spacing)


# Transform axes to voxel space
x_voxel = phys_to_voxel(x_ref)
y_voxel = phys_to_voxel(y_ref)
z_voxel = phys_to_voxel(z_ref)

# Normalize directions
x_voxel_norm = x_voxel / np.linalg.norm(x_voxel)
z_voxel_norm = z_voxel / np.linalg.norm(z_voxel)

# Convert to numpy arrays and reorder to X x Y x Z
US_np = sitk.GetArrayFromImage(US_sitk).transpose(2, 1, 0)
US_mask_np = sitk.GetArrayFromImage(US_mask_sitk).transpose(2, 1, 0)


# ============================================================================
# BLANK OUT HALF OF VOLUME PERPENDICULAR TO BEAM
# ============================================================================

# Convert to torch tensors
US_mask_t = torch.tensor(US_mask_np > 0, dtype=torch.float32)
US_torch = torch.tensor(US_np, dtype=torch.float32)

# Create coordinate grid
nx, ny, nz = US_np.shape
xx, yy, zz = torch.meshgrid(
    torch.arange(nx, dtype=torch.float32),
    torch.arange(ny, dtype=torch.float32),
    torch.arange(nz, dtype=torch.float32),
    indexing="ij"
)
coords = torch.stack((xx, yy, zz), dim=-1)  # (nx, ny, nz, 3)

# Define directions
z_dir = torch.tensor(z_voxel_norm, dtype=torch.float32)  # beam direction
x_dir = torch.tensor(x_voxel_norm, dtype=torch.float32)  # lateral direction

# Compute lateral position for each voxel
lateral_map = coords @ x_dir  # (nx, ny, nz)

# Find the center of the lateral extent within the mask
masked_lateral = lateral_map[US_mask_t > 0]
lateral_center = masked_lateral.mean()

# Create blanking mask based on selected side
if BLANK_SIDE == 'right':
    blank_mask = lateral_map >= lateral_center
elif BLANK_SIDE == 'left':
    blank_mask = lateral_map < lateral_center
else:
    raise ValueError(f"BLANK_SIDE must be 'right' or 'left', got '{BLANK_SIDE}'")

# Apply blanking
US_blanked = US_torch.clone()
US_blanked[blank_mask] = 0.0


# ============================================================================
# VISUALIZATION
# ============================================================================

# Select slice for visualization (perpendicular to lateral direction)
slice_idx = min(US_np.shape[1] // 2, US_np.shape[1] - 1)

# Extract slices
original_slice = US_torch[:, slice_idx, :].cpu().numpy()
blanked_slice = US_blanked[:, slice_idx, :].cpu().numpy()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original_slice.T, cmap='gray', origin='lower')
axes[0].set_title('Original US', fontsize=14)
axes[0].axis('off')

axes[1].imshow(blanked_slice.T, cmap='gray', origin='lower')
axes[1].set_title(f'Blanked ({BLANK_SIDE.capitalize()} Side Removed)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show(block=True)


# ============================================================================
# SAVE OUTPUT
# ============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

# Convert back to SimpleITK image
US_blanked_np = US_blanked.cpu().numpy().transpose(2, 1, 0)  # Back to Z x Y x X
US_blanked_sitk = sitk.GetImageFromArray(US_blanked_np)
US_blanked_sitk.CopyInformation(US_sitk)

# Save
output_path = os.path.join(OUT_DIR, f"US_blanked_{BLANK_SIDE}.nrrd")
sitk.WriteImage(US_blanked_sitk, output_path)
print(f"Saved blanked volume to: {output_path}")