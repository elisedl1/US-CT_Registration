import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# Parameters
blur_sigma = 1.0       # controls blurriness
noise_std = 0.2       # controls speckle intensity

# Paths
input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/moving.nrrd"
output_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3"

# Load the image
img = sitk.ReadImage(input_path)
img_array = sitk.GetArrayFromImage(img).astype(np.float32)  

# Add Gaussian blur to the whole volume
blurred_array = gaussian_filter(img_array, sigma=blur_sigma)

# Add Gaussian noise (speckle) to the whole volume
noisy_array = blurred_array + np.random.normal(0, noise_std, blurred_array.shape)

# Convert back to SimpleITK image
noisy_img = sitk.GetImageFromArray(noisy_array)
noisy_img.CopyInformation(img)  # keep original spacing, origin, direction

# Save the noisy image
output_path = os.path.join(output_dir, f"moving_{blur_sigma}_{noise_std}.nrrd")
sitk.WriteImage(noisy_img, output_path)
print(f"Saved noisy image to: {output_path}")

# Optional: plot mid slice for verification
mid_slice_idx = img_array.shape[0] // 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Mid Slice")
plt.imshow(img_array[mid_slice_idx], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Blurred + Speckle Noise Mid Slice\n(sigma={blur_sigma}, noise={noise_std})")
plt.imshow(noisy_array[mid_slice_idx], cmap='gray')
plt.axis('off')
plt.show()
