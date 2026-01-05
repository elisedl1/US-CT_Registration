import os
import re
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree


def load_mesh(path):
    mesh = pv.read(path)
    mesh = mesh.extract_surface().triangulate().clean()
    return mesh


def uniform_sample(mesh, n_points=50000):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    verts = mesh.points

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
    tri_prob = tri_areas / tri_areas.sum()

    tri_indices = np.random.choice(len(faces), size=n_points, p=tri_prob)

    v0 = v0[tri_indices]
    v1 = v1[tri_indices]
    v2 = v2[tri_indices]

    u = np.random.rand(n_points, 1)
    v = np.random.rand(n_points, 1)

    swap = (u + v > 1)
    u[swap] = 1 - u[swap]
    v[swap] = 1 - v[swap]

    samples = v0 + u * (v1 - v0) + v * (v2 - v0)
    return samples


def parse_level(filename):
    m = re.search(r"L(\d+)", filename)
    if m is None:
        raise ValueError(f"Cannot parse vertebral level from {filename}")
    return int(m.group(1))



def compute_adjacent_vertebra_pairings(
    mesh_dir,
    n_sample=30000,
    n_pairs=500,
    max_dist=8.0,
    seed=42,
    save=False,
    out_dir=None
):


    np.random.seed(seed)

    # load and sort meshes
    files = [
        f for f in os.listdir(mesh_dir)
        if f.endswith(".vtk") and re.search(r"L\d+", f)
    ]

    levels = [(parse_level(f), f) for f in files]
    levels.sort(key=lambda x: x[0])  

    meshes = {}
    for lvl, fname in levels:
        path = os.path.join(mesh_dir, fname)
        meshes[lvl] = load_mesh(path)

    pairings = {}

    # compute pairings for adjacent vertebrae
    for (lvl_i, _), (lvl_j, _) in zip(levels[:-1], levels[1:]):

        print(f"\nComputing pairings: L{lvl_i} → L{lvl_j}")

        mesh_i = meshes[lvl_i]
        mesh_j = meshes[lvl_j]

        pts_i = uniform_sample(mesh_i, n_points=n_sample)
        pts_j = uniform_sample(mesh_j, n_points=n_sample)

        tree_j = cKDTree(pts_j)
        dists, idx = tree_j.query(pts_i)

        valid_idx = np.where(dists < max_dist)[0]

        if len(valid_idx) < n_pairs:
            raise RuntimeError(
                f"L{lvl_i}-L{lvl_j}: only {len(valid_idx)} pairs < {max_dist} mm"
            )

        keep = np.random.choice(valid_idx, size=n_pairs, replace=False)

        pairs_i = pts_i[keep]
        pairs_j = pts_j[idx[keep]]
        d0 = dists[keep]

        pairings[(lvl_i, lvl_j)] = {
            "L_i": pairs_i,
            "L_j": pairs_j,
            "d0": d0
        }

        print(
            f"  kept {n_pairs} pairs | "
            f"min={d0.min():.2f}  mean={d0.mean():.2f}  max={d0.max():.2f} mm"
        )

    return pairings


if __name__ == "__main__":

    mesh_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped"

    pairings = compute_adjacent_vertebra_pairings(
        mesh_dir,
        n_sample=30000,
        n_pairs=500,
        max_dist=8.0,
        seed=42
    )