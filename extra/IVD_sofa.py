import os
import re
import json
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

def load_mesh(path):
    mesh = pv.read(path)
    mesh = mesh.extract_surface().triangulate().clean()
    mesh.compute_normals(inplace=True, auto_orient_normals=True)
    return mesh

def sample_normals(mesh, sampled_points):
    verts = mesh.points
    normals = mesh.point_normals
    tree = cKDTree(verts)
    _, idx = tree.query(sampled_points)
    return normals[idx]

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

def filter_outlier_pairs(dists, pairs_i, pairs_j, k=2.5):
    median = np.median(dists)
    mad = np.median(np.abs(dists - median))
    lower = median - k * mad
    upper = median + k * mad
    keep_idx = np.where((dists >= lower) & (dists <= upper))[0]
    return pairs_i[keep_idx], pairs_j[keep_idx], dists[keep_idx]

def compute_adjacent_vertebra_pairings(mesh_dir, suffix, n_sample=30000, n_pairs=500,
                                       max_dist=8.0, seed=42, outlier_k=2.5):

    np.random.seed(seed)

    # find meshes
    files = [f for f in os.listdir(mesh_dir) if f.endswith(suffix) and re.search(r"L\d+", f)]
    levels = [(parse_level(f), f) for f in files]
    levels.sort(key=lambda x: x[0])
    print("Found vertebra levels:", [lvl for lvl, _ in levels])

    meshes = {lvl: load_mesh(os.path.join(mesh_dir, fname)) for lvl, fname in levels}

    pairings = {}

    for (lvl_i, fname_i), (lvl_j, fname_j) in zip(levels[:-1], levels[1:]):
        mesh_i = meshes[lvl_i]
        mesh_j = meshes[lvl_j]

        pts_i = uniform_sample(mesh_i, n_points=n_sample)
        pts_j = uniform_sample(mesh_j, n_points=n_sample)

        tree_j = cKDTree(pts_j)
        dists, idx = tree_j.query(pts_i)

        normals_i = sample_normals(mesh_i, pts_i)
        normals_j = sample_normals(mesh_j, pts_j)

        dot = np.einsum("ij,ij->i", normals_i, normals_j[idx])
        valid_idx = np.where((dists < max_dist) & (dot < -0.7))[0]

        if len(valid_idx) < n_pairs:
            raise RuntimeError(f"L{lvl_i}-L{lvl_j}: only {len(valid_idx)} valid pairs < {max_dist} mm")

        keep = np.random.choice(valid_idx, size=n_pairs, replace=False)

        # filter outliers
        L_i_filtered, L_j_filtered, d_filtered = filter_outlier_pairs(
            dists[keep], pts_i[keep], pts_j[idx[keep]], k=outlier_k
        )

        # store as list of dictionaries
        pairings[f"L{lvl_i}-L{lvl_j}"] = [
            {"i": list(L_i_filtered[k]), "j": list(L_j_filtered[k]), "d0": float(d_filtered[k])}
            for k in range(len(L_i_filtered))
        ]

        print(f"L{lvl_i}-L{lvl_j}: {len(L_i_filtered)} pairs saved.")

    return pairings, meshes

def visualize_pairings(mesh_i, mesh_j, pairs, subsample=50):
    pairs_i = np.array([p["i"] for p in pairs])
    pairs_j = np.array([p["j"] for p in pairs])

    if len(pairs_i) > subsample:
        idx = np.random.choice(len(pairs_i), subsample, replace=False)
        pairs_i = pairs_i[idx]
        pairs_j = pairs_j[idx]

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_i, color="lightgray", opacity=0.3)
    plotter.add_mesh(mesh_j, color="lightblue", opacity=0.3)
    plotter.add_points(pairs_i, color="red", point_size=8, render_points_as_spheres=True)
    plotter.add_points(pairs_j, color="blue", point_size=8, render_points_as_spheres=True)

    for p, q in zip(pairs_i, pairs_j):
        plotter.add_mesh(pv.Line(p, q), color="yellow", line_width=2)

    plotter.show()
    plotter.close()
    del plotter

if __name__ == "__main__":
    mesh_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/original"
    suffix = "_facet.vtk"
    output_json = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/original/facet_point_pairs.json"

    pairings, meshes = compute_adjacent_vertebra_pairings(
        mesh_dir,
        suffix,
        n_sample=30000,
        n_pairs=500,
        max_dist=8.0,
        seed=42,
        outlier_k=2.5
    )

    # Save all pairs to JSON
    with open(output_json, "w") as f:
        json.dump(pairings, f, indent=4)
    print(f"Saved all vertebra pairs to {output_json}")

    # Optional: visualize each pair
    for pair_name, pairs in pairings.items():
        lvl_i, lvl_j = map(int, re.findall(r"\d+", pair_name))
        print(f"Visualizing {pair_name} ({len(pairs)} pairs)")
        visualize_pairings(meshes[lvl_i], meshes[lvl_j], pairs, subsample=50)
