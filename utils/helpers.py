import torch
import json
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk


def normalize_hessian(hessian: torch.tensor) -> torch.Tensor:
    hessian_magnitude = torch.sum(hessian ** 2, axis=(-1, -2)) ** 0.5
    hessian_magnitude[hessian_magnitude == 0.0] = 1.0
    return hessian / hessian_magnitude.unsqueeze(-1).unsqueeze(-1)


def normalize_grad(grad: torch.tensor) -> torch.tensor:
    grad_magnitude = torch.sum(grad ** 2, axis=-1) ** 0.5
    grad_magnitude[grad_magnitude == 0.0] = 1.0
    return grad / grad_magnitude.unsqueeze(-1)


def create_image_mask(
    image: torch.tensor, dilation_size: int
) -> torch.Tensor:
    bg_tensor = image == 0.0
    bg = bg_tensor.numpy()
    hole_size = 5
    bg = binary_erosion(bg, iterations=hole_size)
    bg = binary_dilation(bg, iterations=hole_size)
    bg = binary_dilation(bg, iterations=dilation_size)
    return torch.tensor(~bg)



def euler_matrix(rx, ry, rz, t):
    # Compute rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R, np.array(t)


def sitk_euler_to_matrix(tx):
    # Extract rotation as a direction cosine matrix
    R = np.array(tx.GetMatrix()).reshape(3, 3)

    # Extract translation
    t = np.array(tx.GetTranslation())

    # Extract center
    c = np.array(tx.GetCenter())

    # Compute: x' = R (x - c) + t + c
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t + c - R @ c
    return M


def transform_affine_3d(
        points: torch.Tensor,
        affine_matrix: torch.Tensor,
        force_rigid: bool = False
) -> torch.Tensor:
    if not force_rigid:
        transformed_points = torch.einsum(
            'ij,...j->...i', affine_matrix[:, :-1].double(), points.double()
        )
        return (transformed_points + affine_matrix[:, -1]).float()

    u, _, v = torch.linalg.svd(affine_matrix[:, :-1] + torch.eye(3))
    if torch.linalg.det(u @ v) < 0:
        u[:, -1] *= -1
    rotation_matrix = u @ v - torch.eye(3)
    rotation_matrix = rotation_matrix.double()
    transformed_points = torch.einsum(
        'ij,...j->...i', rotation_matrix, points.double()
    )
    return (transformed_points + affine_matrix[:, -1]).float()


def compute_determinant(
    gradient: torch.Tensor, hessian: torch.Tensor
) -> torch.Tensor:
    dyadic = torch.einsum('...i,...j->...ij', *2*(gradient,))
    xx = torch.sum(hessian * hessian, axis=(-1, -2))
    yy = torch.sum(dyadic * dyadic, axis=(-1, -2))
    xy = torch.sum(hessian * dyadic, axis=(-1, -2))
    return xx*yy - xy**2


def compute_vector_determinant(
    vector1: torch.Tensor, vector2:torch.Tensor
) -> torch.Tensor:
    xx = torch.sum(vector1 * vector1, axis=-1)
    yy = torch.sum(vector2 * vector2, axis=-1)
    xy = torch.sum(vector1 * vector2, axis=-1)
    return xx*yy - xy**2

def export_samples_to_slicer_json(
    fixed_parser, fixed_mask_indices, output_path, samples_count=1000, name="Fiducials"
):
    """
    Export randomly sampled voxels (within mask) to Slicer Fiducial JSON.
    Converts voxel indices -> LPS coordinates (mm) using parser affine.
    """

    # randomly choose voxel indices
    samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))
    sampled_voxels = fixed_mask_indices[samples]

    # convert voxel indices -> physical coordinates (LPS mm)
    sampled_points = fixed_parser.compute_positions(sampled_voxels)

    # build fiducial control points
    control_points = []
    for idx, point in enumerate(sampled_points, start=1):
        cp = {
            "id": str(idx),
            "label": f"{name}_{idx}",
            "description": "",
            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
            "position": [float(point[0]), float(point[1]), float(point[2])],
            "orientation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        }
        control_points.append(cp)

    slicer_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "name": name,
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": True,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": samples_count,
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5, 0.5],
                    "activeColor": [0.4, 1.0, 0.0],
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0
                }
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(slicer_json, f, indent=4)

    print(f"Saved {samples_count} samples (in LPS mm coords) to {output_path}")


def monogenic_filter(img, scale=2):
    # returns 

    import numpy as np
from scipy.fft import fftn, ifftn, fftshift


def monogenic_filter_3d(img, scale=2):
    """
    Apply monogenic filter
    => amplitude measure local signal strength 
    => intensity -> normalized local energy
    
    Returns:
        A: amplitude
        O_x, O_y, O_z: orientation components
    """
    H, W, D = img.shape
    
    # fourier transform
    F = fftn(img)
    
    # frequency grids
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    w = np.fft.fftfreq(D)
    U, V, W_ = np.meshgrid(u, v, w, indexing='xy')
    
    # riesz transform filters
    magnitude = np.sqrt(U**2 + V**2 + W_**2) + 1e-15
    R_x = 1j * U / magnitude
    R_y = 1j * V / magnitude
    R_z = 1j * W_ / magnitude
    
    # gaussian bandpass / scale
    radius = magnitude
    bandpass = np.exp(-(radius**2)/(2*scale**2))
    
    # apply filters
    F_Rx = F * R_x * bandpass
    F_Ry = F * R_y * bandpass
    F_Rz = F * R_z * bandpass
    
    r_x = np.real(ifftn(F_Rx))
    r_y = np.real(ifftn(F_Ry))
    r_z = np.real(ifftn(F_Rz))
    
    # amplitude
    A = np.sqrt(img**2 + r_x**2 + r_y**2 + r_z**2)
    
    # orientation components
    O_x = r_x / (A + 1e-15)
    O_y = r_y / (A + 1e-15)
    O_z = r_z / (A + 1e-15)
    
    return A, O_x, O_y, O_z


def mean_curvature_3d(volume: torch.Tensor, voxel_size) -> torch.Tensor:
    # gradients (first derivatives)
    f_y, f_x, f_z = torch.gradient(volume, spacing=voxel_size)
    
    # second derivatives
    f_xx, f_xy, f_xz = torch.gradient(f_x, spacing=voxel_size)
    _,   f_yy, f_yz = torch.gradient(f_y, spacing=voxel_size)
    _,   _,   f_zz = torch.gradient(f_z, spacing=voxel_size)
        
    # numerator of mean curvature
    numerator = (
        (1 + f_y**2 + f_z**2) * f_xx +
        (1 + f_x**2 + f_z**2) * f_yy +
        (1 + f_x**2 + f_y**2) * f_zz -
        2*(f_x*f_y*f_xy + f_x*f_z*f_xz + f_y*f_z*f_yz)
    )
    
    # denominator
    denominator = 2 * (f_x**2 + f_y**2 + f_z**2 + 1e-15)**(3/2)
    
    H = numerator / denominator
    return H

def hessian_determinant(hessian: torch.Tensor) -> torch.Tensor:
    H, W, D, _, _ = hessian.shape
    hess_flat = hessian.reshape(-1, 3, 3)
    
    # compute determinant per voxel
    det_flat = torch.linalg.det(hess_flat)
    
    # reshape back to volume
    det_vol = det_flat.reshape(H, W, D)
    return det_vol


def smooth(array: np.ndarray, spacing: tuple, scale: float, remove_background: bool = False) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a 3D volume.

    array: input volume as NumPy array (X x Y x Z)
    spacing: voxel spacing (dx, dy, dz)
    scale: standard deviation of Gaussian in physical units
    remove_background: if True, background (0) voxels remain 0
    """
    arr = array.copy()
    if remove_background:
        bg = arr == 0.0
    # Convert scale to voxel units
    sigma_voxels = np.array([scale / s for s in spacing])
    arr = gaussian_filter(arr, sigma=sigma_voxels)
    if remove_background:
        arr[bg] = 0.0
    return torch.tensor(arr, dtype=torch.float32)


def save_torch_as_nrrd(arr: torch.Tensor, ref_img: sitk.Image, out_file: str):
    """
    Save a PyTorch tensor as a NRRD using a reference SimpleITK image for spacing, origin, etc.
    Works for 3D or 4D arrays (e.g., gradient) or 5D arrays (Hessian).
    """
    arr_np = arr.detach().cpu().numpy()  # in case it's a tensor
    
    # Determine number of dimensions
    if arr_np.ndim == 3:
        # regular volume
        arr_np = arr_np.transpose(2,1,0)  # X,Y,Z -> Z,Y,X
    elif arr_np.ndim == 4:
        # gradient volume: Z,Y,X,3 -> 3,Z,Y,X ?
        arr_np = arr_np.transpose(2,1,0,3)  # reorder only spatial axes
    elif arr_np.ndim == 5:
        # Hessian volume: Z,Y,X,3,3
        arr_np = arr_np.transpose(2,1,0,3,4)
    else:
        raise ValueError(f"Unsupported array shape {arr_np.shape}")
    
    sitk_arr = sitk.GetImageFromArray(arr_np)
    sitk_arr.SetSpacing(ref_img.GetSpacing())
    sitk_arr.SetOrigin(ref_img.GetOrigin())
    sitk_arr.SetDirection(ref_img.GetDirection())
    
    sitk.WriteImage(sitk_arr, out_file)
    print(f"Saved {out_file}")