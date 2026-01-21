import torch
import json
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk


def compute_ivd_collision_loss(pairings, transforms_list, case_names):
    # IVD collision loss tuned for your specific anatomy
    total_loss = 0.0
    metrics = {}
    
    for i in range(len(case_names) - 1):
        vert1 = case_names[i]
        vert2 = case_names[i + 1]
        pair_key = (i + 1, i + 2)
        
        if pair_key not in pairings:
            print(f"SKIPPED - {pair_key} not found in pairings")
            continue
            
        # point pairs for this IVD
        pts1 = pairings[pair_key]['L_i'] 
        pts2 = pairings[pair_key]['L_j']
        initial_distances = pairings[pair_key]['d0']
        
        # transform points to fixed space
        tx1 = transforms_list[i]
        tx2 = transforms_list[i + 1]
        pts1_transformed = np.array([tx1.TransformPoint(p.tolist()) for p in pts1])
        pts2_transformed = np.array([tx2.TransformPoint(p.tolist()) for p in pts2])
        

        # compute current pairwise distances
        current_distances = np.linalg.norm(pts1_transformed - pts2_transformed, axis=1)

        # if pair_key == (2, 3):
        #     d0 = initial_distances
        #     d = current_distances
        #     min_allowed = (1.0 - max_compression_frac) * d0

        #     print("\n=== L2–L3 SPACING DIAGNOSTICS ===")
        #     print(f"initial mean: {np.mean(d0):.2f} mm")
        #     print(f"initial min : {np.min(d0):.2f} mm")
        #     print(f"current mean: {np.mean(d):.2f} mm")
        #     print(f"current min : {np.min(d):.2f} mm")
        #     print(f"min_allowed mean: {np.mean(min_allowed):.2f} mm")
        #     print(f"min_allowed min : {np.min(min_allowed):.2f} mm")

        #     print(f"#points below min_allowed: {np.sum(d < min_allowed)} / {len(d)}")
        #     print(f"max violation (mm): {np.max(np.maximum(0, min_allowed - d)):.2f}")


        # LOSS COMPONENT #1: DIRECTION VECTOR FLIPPING (collision)
        # cosine similiarity between current vectors and v0
        v0 = pairings[pair_key]['v0']  # precomputed initial vectors
        v_current = pts2_transformed - pts1_transformed

        # normalize
        v0_norm = v0 / (np.linalg.norm(v0, axis=1, keepdims=True) + 1e-8)
        v_current_norm = v_current / (np.linalg.norm(v_current, axis=1, keepdims=True) + 1e-8)

        # cosine similarity
        cos_sim = np.einsum("ij,ij->i", v0_norm, v_current_norm)

        # directional penalty if vectors flip (cos_sim < 0)
        flip_idx = cos_sim < 0
        direction_penalty = np.sum(np.exp(-cos_sim[flip_idx]) - 1.0)



        # LOSS COMPONENT 2: COLLISION AVOIDANCE (HARD)
        collision_threshold = 1.0  # mm - hard collision boundary
        violations = collision_threshold - current_distances
        violations = np.maximum(0, violations)
        
        # exp penalty - gets severe as penetration deepens
        collision_penalty = np.sum(np.exp(violations) - 1.0)
        n_collisions = np.sum(current_distances < collision_threshold)
        


        # LOSS COMPONENT 3a: PRESERVE MIN SPACING (SOFT)
        max_compression_frac = 0.35 # percentile
        min_allowed = (1.0 - max_compression_frac) * initial_distances

        spacing_violations = min_allowed - current_distances
        spacing_violations = np.maximum(0, spacing_violations)

        relative_spacing_penalty = np.mean(spacing_violations ** 2) # try .sum?



        # LOSS COMPONENT 3b: PRESERVE MEAN SPACING (SOFT)
        initial_mean = np.mean(initial_distances)
        current_mean = np.mean(current_distances)
        
        # tolerance
        tolerance = 1.5  # mm - allows curvature/natural compression
        mean_deviation = abs(current_mean - initial_mean) - tolerance
        
        if mean_deviation > 0:
            mean_spacing_penalty = mean_deviation ** 2
        else:
            mean_spacing_penalty = 0.0



        # COMPUTE LOSS
        w_collision = 1.0   # critical - prevent collisions, mayeb try 2?
        w_mean_spacing = 0.5   # moderate - maintain anatomy
        w_direction = 1.0
        w_relative_spacing = 1.0
        
        pair_loss = (
            w_collision * collision_penalty +
            w_relative_spacing * relative_spacing_penalty +
            w_mean_spacing * mean_spacing_penalty +
            w_direction * direction_penalty
        )
        
        total_loss += pair_loss

        pair_name = f"L{pair_key[0]}-L{pair_key[1]}"
        
        # store metrics
        metrics[pair_name] = {
            'pair_idx': pair_key,   
            'current_mean': float(current_mean),
            'initial_mean': float(initial_mean),
            'current_min': float(np.min(current_distances)),
            'current_max': float(np.max(current_distances)),
            'current_std': float(np.std(current_distances)),
            'n_collisions': int(n_collisions),
            'collision_loss': float(w_collision * collision_penalty),
            'mean_spacing_loss': float(w_mean_spacing * mean_spacing_penalty),
            'total_loss': float(pair_loss)
        }
    
    return total_loss, metrics


def compute_inter_vertebral_displacement_penalty(moved_centroids, case_centroids, case_axes, transforms_list, margins):

    # constraint relative displacement between neighboring vertebrae along anatomical axes.
    # compares each movement in its transformed anatomical frame

    penalty = 0.0 # start at 0
    K = len(moved_centroids)
    
    if K < 2:
        return 0.0

    # compte movement in transform anatomical frame
    LM_axis_shared = case_axes[0][0]  # same for all vertebrae
    movements_LM = []
    movements_AP = []
    movements_SI = []
    
    for k in range(K - 1):
        
        # rel position in TRANSFORMED space
        relative_vec = moved_centroids[k+1] - moved_centroids[k]
        
        # Transform the vertebra k
        tx_k = transforms_list[k]
        rotation_k = np.array(tx_k.GetMatrix()).reshape(3, 3)
        LM_axis_k_orig, AP_axis_k_orig, SI_axis_k_orig = case_axes[k]
        LM_axis_k_transformed = rotation_k @ LM_axis_k_orig
        AP_axis_k_transformed = rotation_k @ AP_axis_k_orig
        SI_axis_k_transformed = rotation_k @ SI_axis_k_orig
        
        # Vertebra k+1
        tx_k1 = transforms_list[k+1]
        rotation_k1 = np.array(tx_k1.GetMatrix()).reshape(3, 3)
        LM_axis_k1_orig, AP_axis_k1_orig, SI_axis_k1_orig = case_axes[k+1]
        LM_axis_k1_transformed = rotation_k1 @ LM_axis_k1_orig
        AP_axis_k1_transformed = rotation_k1 @ AP_axis_k1_orig
        SI_axis_k1_transformed = rotation_k1 @ SI_axis_k1_orig
        
        # average the transformed axes (compromise reference frame)
        LM_axis_avg = (LM_axis_k_transformed + LM_axis_k1_transformed) / 2.0
        LM_axis_avg = LM_axis_avg / np.linalg.norm(LM_axis_avg)
        
        AP_axis_avg = (AP_axis_k_transformed + AP_axis_k1_transformed) / 2.0
        AP_axis_avg = AP_axis_avg / np.linalg.norm(AP_axis_avg)
        
        SI_axis_avg = (SI_axis_k_transformed + SI_axis_k1_transformed) / 2.0
        SI_axis_avg = SI_axis_avg / np.linalg.norm(SI_axis_avg)


        # SI TWIST PENALTY
        # take LM axes (could also use AP)
        LM_k = LM_axis_k_transformed
        LM_k1 = LM_axis_k1_transformed

        LM_k_proj = LM_k - np.dot(LM_k, SI_axis_avg) * SI_axis_avg
        LM_k1_proj = LM_k1 - np.dot(LM_k1, SI_axis_avg) * SI_axis_avg

        LM_k_proj /= np.linalg.norm(LM_k_proj)
        LM_k1_proj /= np.linalg.norm(LM_k1_proj)

        # project relative position onto averaged transformed axes
        LM_component = abs(np.dot(relative_vec, LM_axis_avg))
        AP_component = abs(np.dot(relative_vec, AP_axis_avg))
        SI_component = abs(np.dot(relative_vec, SI_axis_avg))

        # angle between projected LM axes = twist about SI
        cos_theta = np.clip(np.dot(LM_k_proj, LM_k1_proj), -1.0, 1.0)
        twist_angle = np.arccos(cos_theta)  
        SI_rot_violation = max(0.0, twist_angle - margins['SI_rot'])
        

        # penalize if lateral or AP separation is too large (anatomically implausible)
        LM_magnitude = abs(LM_component)
        AP_magnitude = abs(AP_component)
        
        LM_violation = max(0.0, LM_magnitude - margins['LM'])
        AP_violation = max(0.0, AP_magnitude - margins['AP'])
        
        # For SI: check if compression/extension is excessive
        # expect some SI separation, so check change from original
        original_relative_vec = case_centroids[k+1] - case_centroids[k]
        original_SI = abs(np.dot(original_relative_vec, SI_axis_avg))
        SI_change = abs(abs(SI_component) - original_SI)
        SI_violation = max(0.0, SI_change - margins['SI'])

        penalty += (
            LM_violation**2 +
            AP_violation**2 +
            SI_violation**2 +
            SI_rot_violation**2
        )
    
    return penalty / float(K - 1)




def make_hinge_axes(json_path):

    # load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # extract points
    control_points = data['markups'][0]['controlPoints']
    points = {p['label']: np.array(p['position']) for p in control_points}

    point1 = points['HingeCenter']
    point2 = points['DiscNormEnd']
    point3 = points['HingeNormEnd']
    
    # compute hinge axis
    hinge_axis = (point2 - point1)
    hinge_axis /= np.linalg.norm(hinge_axis) 

    # compute disc axis
    disc_axis = (point3 - point1)
    disc_axis /= np.linalg.norm(disc_axis) 

    return hinge_axis, disc_axis


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

def create_boundary_exclusion_mask(image: torch.Tensor, erosion_size: int) -> torch.Tensor:
    """
    Create a mask that excludes a margin along the US boundary.
    - image: 3D US volume (torch.Tensor)
    - erosion_size: number of voxels to exclude from boundary
    """
    # 1. Background: voxels with value 0
    bg = (image == 0).cpu().numpy()  # boolean

    # 2. Dilate background inwards to exclude boundary voxels
    # binary_dilation expands True values (background) into the foreground
    bg_dilated = binary_dilation(bg, iterations=erosion_size)

    # 3. Invert: True = inside US but away from boundary
    mask = ~bg_dilated

    return torch.tensor(mask, dtype=torch.bool, device=image.device)


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
    """Convert SimpleITK Euler3DTransform -> 4x4 numpy affine matrix."""
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