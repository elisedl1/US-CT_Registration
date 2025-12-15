import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from scipy.ndimage import binary_erosion

def extract_surface(vol):
    arr = sitk.GetArrayFromImage(vol)
    labels = np.unique(arr)
    labels = labels[labels != 0]
    if len(labels) == 0:
        raise ValueError("No non-zero label found in volume.")
    label = labels[0]
    arr = arr == label
    kernel = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(arr, structure=kernel)
    surface = arr & (~eroded)
    out = sitk.GetImageFromArray(surface.astype(np.uint8))
    out.CopyInformation(vol)
    return out

if __name__ == "__main__":
    root = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases"

    paths = sorted(glob(os.path.join(root, "L*", "CT_*_downsampled.nrrd")))

    for path in paths:
        vol = sitk.ReadImage(path)
        surf = extract_surface(vol)
        out_dir = os.path.dirname(path)
        base = os.path.basename(path).replace("_downsampled.nrrd", "_surface.nrrd")
        sitk.WriteImage(surf, os.path.join(out_dir, base))
