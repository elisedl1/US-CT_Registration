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

    samples = v0 + u*(v1 - v0) + v*(v2 - v0)
    return samples


def build_connection_lines(points_A, points_B_nearest, mask):
    points_A_masked = points_A[mask]
    points_B_masked = points_B_nearest[mask]

    n = len(points_A_masked)
    all_points = np.vstack([points_A_masked, points_B_masked])

    lines = []
    for i in range(n):
        lines.append(2)
        lines.append(i)
        lines.append(i + n)

    lines = np.array(lines)
    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = lines
    return poly


if __name__ == "__main__":

    file_L1 = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/L1_body.vtk"
    file_L2 = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/cropped/L2_body.vtk"

    print("Loading meshes...")
    mesh_L1 = load_mesh(file_L1)
    mesh_L2 = load_mesh(file_L2)

    print("Sampling points...")
    pts_L1 = uniform_sample(mesh_L1, n_points=5000)
    pts_L2 = uniform_sample(mesh_L2, n_points=5000)

    tree_L2 = cKDTree(pts_L2)
    d_A2B, idx_A2B = tree_L2.query(pts_L1)
    mask_A = d_A2B < 5.0
    pts_A2B = pts_L2[idx_A2B]

    tree_L1 = cKDTree(pts_L1)
    d_B2A, idx_B2A = tree_L1.query(pts_L2)
    mask_B = d_B2A < 5.0
    pts_B2A = pts_L1[idx_B2A]

    chamfer_sym = (np.mean(d_A2B[mask_A]) + np.mean(d_B2A[mask_B])) / 2
    print(f"Symmetric Chamfer distance (<5mm points): {chamfer_sym:.3f} mm")
    print(f"Points L1 → L2 under 5mm: {np.sum(mask_A)}, Points L2 → L1 under 5mm: {np.sum(mask_B)}")

    connections_forward = build_connection_lines(pts_L1, pts_A2B, mask_A)
    connections_backward = build_connection_lines(pts_L2, pts_B2A, mask_B)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_L1, color="red", opacity=0.3)
    plotter.add_mesh(mesh_L2, color="blue", opacity=0.3)
    plotter.add_mesh(connections_forward, color="yellow", line_width=2)
    plotter.add_mesh(connections_backward, color="green", line_width=2)
    plotter.show()
