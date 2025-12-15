import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import binary_erosion

def resample_to_target(vol, target):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(vol)

def extract_surface(vol):
    arr = sitk.GetArrayFromImage(vol)
    kernel = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(arr, structure=kernel)
    surface = arr & (~eroded)
    surface_img = sitk.GetImageFromArray(surface.astype(np.uint8))
    surface_img.CopyInformation(vol)
    return surface_img

def check_overlap(vol1, vol2):
    vol2_resampled = resample_to_target(vol2, vol1)
    arr1 = sitk.GetArrayFromImage(vol1) > 0
    arr2 = sitk.GetArrayFromImage(vol2_resampled) > 0
    return np.sum(arr1 & arr2)

if __name__ == "__main__":
    vol1_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L1/CT_L1_surface_collision.nrrd"
    vol2_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L2/CT_L2_surface.nrrd"

    vol1 = sitk.ReadImage(vol1_path)
    vol2 = sitk.ReadImage(vol2_path)

    n_overlap = check_overlap(vol1, vol2)
    print(f"Number of overlapping surface voxels: {n_overlap}")
    if n_overlap > 0:
        print("Volumes overlap!")
    else:
        print("No overlap detected.")
