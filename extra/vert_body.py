import SimpleITK as sitk
import numpy as np
from pathlib import Path
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


def get_body(ct_files, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cropped_images = []
    for ct_file in ct_files:
        img = load_image(ct_file)
        centroid = np.array(compute_centroid(ct_file))
        img_cropped = crop_above_centroid_voxel(img, centroid)
        
        # Save the cropped image
        filename = Path(ct_file).name
        output_path = Path(output_dir) / filename
        sitk.WriteImage(img_cropped, str(output_path))
        
        cropped_images.append(img_cropped)
    
    return cropped_images


if __name__ == "__main__":
    CT_files = [
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original_sofa/CT_L1_decimated.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original_sofa/CT_L2_decimated.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original_sofa/CT_L3_decimated.nrrd",
        "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original_sofa/CT_L4_decimated.nrrd",
    ]

    output_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations"
    
    cropped = get_body(CT_files, output_dir)
    
    print("Cropped and saved", len(cropped), "images to", output_dir)