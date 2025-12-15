import os
import numpy as np
import SimpleITK as sitk

def compute_centroid(nrrd_path):
    img = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(img)
    points_voxel = np.argwhere(arr > 0)
    if points_voxel.size == 0:
        raise RuntimeError(f"No segmentation found in {nrrd_path}")
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3,3)
    points_voxel_xyz = points_voxel[:, ::-1]
    points_physical = origin + (direction @ (points_voxel_xyz.T * spacing[:, None])).T
    return points_physical.mean(axis=0)

def process_folder(folder_path):
    vertebrae = ['L1', 'L2', 'L3', 'L4']
    centroids = {}
    for v in vertebrae:
        file_path = os.path.join(folder_path, f'CT_{v}.nrrd')
        if os.path.exists(file_path):
            centroids[v] = compute_centroid(file_path)
    return centroids

def compute_distances(centroids):
    vertebrae = ['L1', 'L2', 'L3', 'L4']
    distances = {}
    for i in range(len(vertebrae)-1):
        v1, v2 = vertebrae[i], vertebrae[i+1]
        if v1 in centroids and v2 in centroids:
            distances[f'{v1}-{v2}'] = np.linalg.norm(centroids[v2] - centroids[v1])
    return distances

folders = {
    "original": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original",
    "intra1_seg": "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg"
}

all_centroids = {}
all_distances = {}

for name, folder in folders.items():
    centroids = process_folder(folder)
    distances = compute_distances(centroids)
    all_centroids[name] = centroids
    all_distances[name] = distances
    print(f"\n{name}")
    for v, c in centroids.items():
        print(f"{v} centroid: {c}")
    for pair, d in distances.items():
        print(f"{pair} distance: {d:.2f}")

print("\nDistance differences (original - intra1_seg):")
for pair in all_distances['original']:
    if pair in all_distances['intra1_seg']:
        diff = abs(all_distances['original'][pair] - all_distances['intra1_seg'][pair])
        print(f"{pair} difference: {diff:.2f}")
