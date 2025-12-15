import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
num_occlusions = 20         # number of rectangles to insert
min_occlusion_size = 10      # minimum rectangle size in voxels
max_occlusion_size = 30     # maximum rectangle size in voxels

# Paths
input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/moving.nrrd"
output_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3"

# Load image
img = sitk.ReadImage(input_path)
img_array = sitk.GetArrayFromImage(img).astype(np.float32)

occluded_array = img_array.copy()

# Add occlusions
np.random.seed(42)
depth, height, width = img_array.shape

for _ in range(num_occlusions):
    # random size for each dimension
    d_size = np.random.randint(min_occlusion_size, max_occlusion_size + 1)
    h_size = np.random.randint(min_occlusion_size, max_occlusion_size + 1)
    w_size = np.random.randint(min_occlusion_size, max_occlusion_size + 1)

    # random position (ensure it fits)
    d_start = np.random.randint(0, max(1, depth - d_size))
    h_start = np.random.randint(0, max(1, height - h_size))
    w_start = np.random.randint(0, max(1, width - w_size))

    # apply occlusion
    occluded_array[d_start:d_start + d_size,
                   h_start:h_start + h_size,
                   w_start:w_start + w_size] = 0

# Convert back to SITK
occluded_img = sitk.GetImageFromArray(occluded_array)
occluded_img.CopyInformation(img)

# Save
output_path = os.path.join(
    output_dir,
    f"moving_occluded_{num_occlusions}_{min_occlusion_size}-{max_occlusion_size}.nrrd"
)
sitk.WriteImage(occluded_img, output_path)
print(f"Saved occluded image to: {output_path}")

# Plot mid-slice for verification
mid_slice_idx = depth // 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Mid Slice")
plt.imshow(img_array[mid_slice_idx], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Occluded Mid Slice\n({num_occlusions} rectangles)")
plt.imshow(occluded_array[mid_slice_idx], cmap='gray')
plt.axis('off')
plt.show()
