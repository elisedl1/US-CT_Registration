import SimpleITK as sitk
import numpy as np
import os

in_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/fixed.nrrd"

# Read image
img = sitk.ReadImage(in_path)
arr = sitk.GetArrayFromImage(img).astype(np.float32)

# 90th percentile
p90 = np.percentile(arr, 90)

# Scale brightest voxels
arr[arr > p90] *= 0.05

# Back to image, preserve metadata
out_img = sitk.GetImageFromArray(arr)
out_img.CopyInformation(img)

# Save
out_path = in_path.replace(".nrrd", "_p90_scaled0p2.nrrd")
sitk.WriteImage(out_img, out_path)

print("Saved:", out_path)


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

orig_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/fixed.nrrd"
mod_path  = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/fixed_p90_scaled0p2.nrrd"

# Load volumes
orig = sitk.GetArrayFromImage(sitk.ReadImage(orig_path))
mod  = sitk.GetArrayFromImage(sitk.ReadImage(mod_path))

# Choose axial slice (middle)
z = orig.shape[0] // 2

# Shared windowing based on original
vmin = np.percentile(orig, 1)
vmax = np.percentile(orig, 99)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(orig[z], cmap = "gray" , vmin=vmin, vmax=vmax)
plt.title("Original (axial)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mod[z], cmap = "gray", vmin=vmin, vmax=vmax)
plt.title("P90 scaled ×0.2 (axial)")
plt.axis("off")

plt.tight_layout()
plt.show()