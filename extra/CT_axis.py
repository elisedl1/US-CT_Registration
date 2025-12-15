import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from centroid import compute_centroid

# Load CT
CT_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L1.nrrd"
ct_image = sitk.ReadImage(CT_file)
# Centroid in world space
centroid_world = np.array(compute_centroid(CT_file))

# Voxel axes in world space
D = np.array(ct_image.GetDirection()).reshape(3,3)

# Axes vectors
LM_axis = D[:,0]
AP_axis = D[:,1]
SI_axis = D[:,2]

# Scale for visualization (mm)
scale = 50.0

# Compute endpoints
LM_point = centroid_world + LM_axis * scale
AP_point = centroid_world + AP_axis * scale
SI_point = centroid_world + SI_axis * scale

print("Centroid (world):", centroid_world)
print("LM endpoint (world):", LM_point)
print("AP endpoint (world):", AP_point)
print("SI endpoint (world):", SI_point)