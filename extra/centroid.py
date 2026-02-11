import numpy as np
import SimpleITK as sitk

def compute_centroid(nrrd_path):
    """
    Compute the centroid of a segmentation NRRD in physical space.
    Returns a numpy array of shape (3,).
    """
    img = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)

    points_voxel = np.argwhere(arr > 0)
    if points_voxel.size == 0:
        raise RuntimeError(f"No segmentation found in {nrrd_path}")

    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3,3)

    points_voxel_xyz = points_voxel[:, ::-1]  # (X,Y,Z)
    points_physical = origin + (direction @ (points_voxel_xyz.T * spacing[:, None])).T

    return points_physical.mean(axis=0)


# path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1_old/Cases/L1/CT_L1.nrrd"
# path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L1/fixed.nrrd"
# path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_seg_combined.nrrd"


# path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L4.nrrd"

# print(compute_centroid(path))

