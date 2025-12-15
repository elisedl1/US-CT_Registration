import SimpleITK as sitk
import numpy as np

def downsample_volume_keep_spacing(volume, factor=4):
    arr = sitk.GetArrayFromImage(volume)
    downsampled_arr = arr[::factor, ::factor, ::factor]
    downsampled_img = sitk.GetImageFromArray(downsampled_arr)
    
    # Adjust spacing according to downsampling factor
    orig_spacing = np.array(volume.GetSpacing())
    downsampled_spacing = orig_spacing * factor
    downsampled_img.SetSpacing(downsampled_spacing.tolist())
    
    # Keep origin and direction
    downsampled_img.SetOrigin(volume.GetOrigin())
    downsampled_img.SetDirection(volume.GetDirection())
    
    return downsampled_img

# Usage
nrrd_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L1/CT_L1.nrrd"
volume = sitk.ReadImage(nrrd_path)
downsampled_volume = downsample_volume_keep_spacing(volume, factor=4)
output_path = nrrd_path.replace(".nrrd", "_downsampled.nrrd")
sitk.WriteImage(downsampled_volume, output_path)
print(f"Downsampled volume saved to {output_path}")
