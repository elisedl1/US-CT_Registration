import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Create mock images
# --------------------------
# CT image: 100x100, a square somewhere else (not centered)
ct_array = np.zeros((100, 100), dtype=np.float32)
ct_array[20:40, 70:90] = 1.0  # square in top-right quadrant
ct_image = sitk.GetImageFromArray(ct_array)

# US image: 100x100, a circle in the center
us_array = np.zeros((100, 100), dtype=np.float32)
xx, yy = np.meshgrid(np.arange(100), np.arange(100))
circle = (xx-50)**2 + (yy-50)**2 < 15**2
us_array[circle] = 1.0
us_image = sitk.GetImageFromArray(us_array)

# --------------------------
# 2. Define a transform
# --------------------------
# Let's just translate 20 pixels right, 10 pixels down
Tx = sitk.TranslationTransform(2)
Tx.SetOffset((20, 10))  # (x, y) offset in physical space

# --------------------------
# 3. Apply transform to both images
# --------------------------
moved_us = sitk.Resample(us_image, us_image, Tx, sitk.sitkLinear, 0.0, us_image.GetPixelID())
moved_ct = sitk.Resample(ct_image, ct_image, Tx, sitk.sitkLinear, 0.0, ct_image.GetPixelID())

# --------------------------
# 4. Convert to numpy for plotting
# --------------------------
us_np = sitk.GetArrayFromImage(us_image)
ct_np = sitk.GetArrayFromImage(ct_image)
moved_us_np = sitk.GetArrayFromImage(moved_us)
moved_ct_np = sitk.GetArrayFromImage(moved_ct)

# --------------------------
# 5. Plot results
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

axes[0, 0].imshow(us_np, cmap='gray')
axes[0, 0].set_title('Original US')
axes[0, 1].imshow(ct_np, cmap='gray')
axes[0, 1].set_title('Original CT')

axes[1, 0].imshow(moved_us_np, cmap='gray')
axes[1, 0].set_title('Moved US')
axes[1, 1].imshow(moved_ct_np, cmap='gray')
axes[1, 1].set_title('Moved CT')

plt.tight_layout()
plt.show()
