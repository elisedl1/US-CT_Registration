import SimpleITK as sitk
import numpy as np
import pyvista as pv
from skimage import measure
import json
from json_save_hinge import save_json_points

ivd_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/hinges/CT_L3_body_CT_L4_body_disc_mask.nrrd"

def compute_centroid(nrrd_path):
    img = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(img)
    points_voxel = np.argwhere(arr > 0)
    if points_voxel.size == 0:
        raise RuntimeError(f"No segmentation found in {nrrd_path}")

    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3,3)

    # Voxel coordinates in (X,Y,Z)
    points_voxel_xyz = points_voxel[:, ::-1]
    points_physical = origin + (direction @ (points_voxel_xyz.T * spacing[:, None])).T
    return points_physical.mean(axis=0)

# Load segmentation
img = sitk.ReadImage(ivd_path)
arr = sitk.GetArrayFromImage(img)
spacing = np.array(img.GetSpacing())
origin = np.array(img.GetOrigin())
direction = np.array(img.GetDirection()).reshape(3,3)

# Extract surface points
verts, faces, _, _ = measure.marching_cubes(arr, level=0.5)

# Correct voxel order for physical conversion
verts_xyz = verts[:, [2,1,0]]  # Z,Y,X -> X,Y,Z
verts_world = origin + (direction @ (verts_xyz * spacing).T).T

# Compute volume centroid (Slicer-consistent)
centroid = compute_centroid(ivd_path)

# PCA relative to volume centroid
points = verts_world - centroid
_, _, vh = np.linalg.svd(points, full_matrices=False)
disc_normal = vh[2] / np.linalg.norm(vh[2])
ap_direction = vh[1]
ml_axis = np.cross(disc_normal, ap_direction)
ml_axis /= np.linalg.norm(ml_axis)

# # PyVista visualization
# cloud = pv.PolyData(verts_world)
# normal_arrow = pv.Arrow(start=centroid, direction=disc_normal, scale=10)
# hinge_arrow = pv.Arrow(start=centroid, direction=ml_axis, scale=10)

# plotter = pv.Plotter()
# plotter.add_mesh(cloud, color="lightblue", point_size=5)
# plotter.add_mesh(normal_arrow, color="red", line_width=4, label="Disc Normal")
# plotter.add_mesh(hinge_arrow, color="yellow", line_width=4, label="Hinge Axis")
# plotter.add_legend()
# plotter.add_axes()
# plotter.show()

print("hinge point =", centroid) # LPS world space
print("disc normal =", disc_normal) # LPS world space
print("flexion/extension hinge axis =", ml_axis) # LPS world space 



# create multiple points for visual validation in slicer
save_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/hinges/L3_L4_hinge.json"
save_json_points(centroid, disc_normal, ml_axis, save_path, scale=10.0, volume_node_id="vtkMRMLScalarVolumeNode6")
