import pyvista as pv
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import SimpleITK as sitk
import os
import json

# ----------------------------------------------------------
# Load mesh (STL → vertices + faces)
# ----------------------------------------------------------
def load_mesh(path):
    tm = trimesh.load(path)
    faces = np.hstack([np.full((len(tm.faces), 1), 3), tm.faces]).astype(np.int64)
    return pv.PolyData(tm.vertices, faces), tm.vertices

# ----------------------------------------------------------
# Read transform from NRRD
# ----------------------------------------------------------
def load_nrrd_transform(nrrd_path):
    img = sitk.ReadImage(nrrd_path)
    direction = np.array(img.GetDirection()).reshape(3,3)
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    return direction, spacing, origin

# ----------------------------------------------------------
# Convert world→voxel (returns N x 3)
# ----------------------------------------------------------
def world_to_voxel(pts, direction, spacing, origin):
    pts_rel = pts - origin
    vox = (np.linalg.inv(direction) @ pts_rel.T).T
    return vox / spacing

# ----------------------------------------------------------
# Farthest point sampling
# ----------------------------------------------------------
def farthest_point_sampling(points, k):
    sampled = [np.random.randint(len(points))]
    for _ in range(1, k):
        d = np.min(np.linalg.norm(points - points[sampled][:, None, :], axis=2), axis=0)
        sampled.append(np.argmax(d))
    return np.array(sampled)

# ----------------------------------------------------------
# Compute close connections between two meshes
# ----------------------------------------------------------
def compute_connections(meshA_path, meshB_path, nrrdA_path, labelA, labelB):
    meshA, ptsA = load_mesh(meshA_path)
    meshB, ptsB = load_mesh(meshB_path)

    treeB = cKDTree(ptsB)
    dist, idx = treeB.query(ptsA)
    thr = np.percentile(dist, 10)
    mask = dist <= thr

    closeA = ptsA[mask]
    closeB = ptsB[idx[mask]]

    k = min(100, len(closeA))
    samp_idx = farthest_point_sampling(closeA, k)
    sampA = closeA[samp_idx]
    sampB = closeB[samp_idx]

    direction, spacing, origin = load_nrrd_transform(nrrdA_path)
    voxelA = world_to_voxel(sampA, direction, spacing, origin)

    # Save both .npy and slicer .json
    out_dir = os.path.dirname(meshA_path)

    np.save(os.path.join(out_dir, f"{labelA}_{labelB}_sampA_world.npy"), sampA)
    np.save(os.path.join(out_dir, f"{labelA}_{labelB}_sampB_world.npy"), sampB)
    np.save(os.path.join(out_dir, f"{labelA}_{labelB}_sampA_voxel.npy"), voxelA)

    save_slicer_markups_json(
        os.path.join(out_dir, f"{labelA}_{labelB}_sampA_world_fiducials.json"),
        sampA,
        label_prefix=f"{labelA}_A"
    )

    save_slicer_markups_json(
        os.path.join(out_dir, f"{labelA}_{labelB}_sampB_world_fiducials.json"),
        sampB,
        label_prefix=f"{labelB}_B"
    )

    save_slicer_markups_json(
        os.path.join(out_dir, f"{labelA}_{labelB}_sampA_voxel_fiducials.json"),
        voxelA,
        label_prefix=f"{labelA}_A_vox"
    )

    return meshA, meshB, closeA, closeB, sampA, sampB

# ----------------------------------------------------------
# Write Slicer Markups JSON (schema v1.0.3)
# ----------------------------------------------------------
def save_slicer_markups_json(path, points, label_prefix, associatedNodeID="vtkMRMLScalarVolumeNode6"):
    control_points = []
    orientation = [-1.0, -0.0, -0.0,
                   -0.0, -1.0, -0.0,
                    0.0,  0.0,  1.0]

    for i, p in enumerate(points, start=1):
        control_points.append({
            "id": str(i),
            "label": f"{label_prefix}_{i}",
            "description": "",
            "associatedNodeID": associatedNodeID,
            "position": [float(p[0]), float(p[1]), float(p[2])],
            "orientation": orientation,
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        })

    markups_block = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(points),
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "visibility": False,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [0.3333333333333333, 0.0, 0.0],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 2.9,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.7,
                    "glyphSize": 3.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "translationHandleVisibility": True,
                    "rotationHandleVisibility": True,
                    "scaleHandleVisibility": False,
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }

    with open(path, "w") as f:
        json.dump(markups_block, f, indent=2)

# ----------------------------------------------------------
# Run across L1–L4
# ----------------------------------------------------------
base = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases"

pairs = [
    ("L1", "L2"),
    ("L2", "L3"),
    ("L3", "L4")
]

results = {}
for A, B in pairs:
    meshA = f"{base}/{A}/{A}.stl"
    meshB = f"{base}/{B}/{B}.stl"
    nrrdA = f"{base}/{A}/fixed.nrrd"
    results[(A, B)] = compute_connections(meshA, meshB, nrrdA, A, B)

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
pl = pv.Plotter()

for (A, B), (mA, mB, closeA, closeB, sampA, sampB) in results.items():
    pl.add_mesh(mA, color='white', opacity=0.15)
    pl.add_mesh(mB, color='gray', opacity=0.15)

    for p1, p2 in zip(sampA, sampB):
        pl.add_mesh(pv.Line(p1, p2), color='green', line_width=3)

pl.show()
