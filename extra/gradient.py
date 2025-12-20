import SimpleITK as sitk
import numpy as np

input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L1/fixed.nrrd"
output_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L1/fixed_grad.nrrd"
us_image = sitk.ReadImage(input_path)
us_array = sitk.GetArrayFromImage(us_image)

# compute gradient with voxel spacing
grad_z, grad_y, grad_x = np.gradient(us_array, *us_image.GetSpacing())
grad_volume = np.stack((grad_z, grad_y, grad_x), axis=-1)
grad_image = sitk.GetImageFromArray(grad_volume, isVector=True)
grad_image.CopyInformation(us_image)
sitk.WriteImage(grad_image, output_path)
print("Gradient saved at:", output_path)
