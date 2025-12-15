import SimpleITK as sitk
import numpy as np
from centroid import compute_centroid


def load_image(nrrd_file):
    return sitk.ReadImage(nrrd_file)


def crop_above_centroid_voxel(img, centroid_world):
    arr = sitk.GetArrayFromImage(img)
    voxel_coords = img.TransformPhysicalPointToContinuousIndex(tuple(centroid_world))
    y_centroid_idx = int(round(voxel_coords[1]))
    arr[:, y_centroid_idx:, :] = 0

    cropped_img = sitk.GetImageFromArray(arr)
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetDirection(img.GetDirection())
    return cropped_img


def get_body(ct_files):
    cropped_images = []
    for ct_file in ct_files:
        img = load_image(ct_file)
        centroid = np.array(compute_centroid(ct_file))
        img_cropped = crop_above_centroid_voxel(img, centroid)
        cropped_images.append(img_cropped)
    return cropped_images


if __name__ == "__main__":
    CT_files = [
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L1.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L2.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L3.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L4.nrrd",
    ]

    cropped = get_body(CT_files)
    print("Cropped", len(cropped), "images")
