import nrrd
import numpy as np
import os
import SimpleITK as sitk
from utils.helpers import (
    smooth,
    save_torch_as_nrrd
)
import torch

# Path to your NRRD file
nrrd_file = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT.nrrd'

# Load the NRRD file
CT_sitk = sitk.ReadImage(nrrd_file)
CT_np = sitk.GetArrayFromImage(CT_sitk)  # Z x Y x X
print(CT_np.shape)
CT_np = CT_np.transpose(2, 1, 0, 3)


# Compute gradient magnitude
grad_magnitude = np.linalg.norm(CT_np, axis=-1)
grad_mag_tensor = torch.tensor(grad_magnitude, dtype=torch.float32)

# Save the gradient magnitude as a new NRRD
out_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT.nrrd"
save_torch_as_nrrd(grad_mag_tensor, CT_sitk, os.path.join(out_dir, "CT_grad_mag_test.nrrd"))
