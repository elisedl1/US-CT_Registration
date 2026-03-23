"""
US Simulation from CT and BCLC2 Similarity Metric
Based on: Gill et al. "Biomechanically constrained groupwise ultrasound to CT
registration of the lumbar spine", Medical Image Analysis 16 (2012) 662-674.

Implements:
  - Equations 1-7:  US simulation from CT via ray-casting
  - Equation 8:     LC2 similarity metric
  - Equations 9-11: Biomechanical stiffness matrix energy
  - Equation 12:    BCLC2 = LC2 - sigma * E

Usage in your main script
--------------------------
from us_simulation import preload_ct_hu_volumes, evaluate_group_gpu_gill

# In run_single_registration, after loading moving_parsers add:
ct_hu_parsers = preload_ct_hu_volumes(cases_dir, case_names, device='cuda')

# Replace partial_eval with:
partial_eval = partial(
    evaluate_group_gpu_gill,
    K=K,
    centers=centers,
    ct_hu_parsers=ct_hu_parsers,
    fixed_parser=fixed_parser,
    case_centroids=case_centroids,
    case_names=case_names,
    max_iter=max_iter,
    sigma=1.0,          # biomechanical model weight  (0, 0.5, 1, 2 per paper)
    beam_axis=1,        # AP axis in DHW layout
    device='cuda',
)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

from utils.file_parser import PyNrrdParser
from utils.helpers import sitk_euler_to_matrix


# ---------------------------------------------------------------------------
# CONSTANTS (from Gill et al.)
# ---------------------------------------------------------------------------
HU_FULL_REFLECTION_THRESHOLD = 250.0   # tau  — Eq. 4
LOG_COMPRESS_ALPHA           = 10.0    # a    — Eq. 5, user-defined
CT_TO_US_SCALE               = 1.36    # Eq. 6
CT_TO_US_OFFSET              = -1429.0 # Eq. 6

# Biomechanical stiffness matrix K (Eq. 9)
# From Desroches et al. (2007), adapted from Panjabi et al. (1976)
# Row/col order: Fx Fy Fz Mx My Mz  <->  Tx Ty Tz Rx Ry Rz
STIFFNESS_MATRIX = np.array([
    [ 100,    0,   50,       0,   -1640,      0],
    [   0,  110,    0,     150,       0,    580],
    [  50,    0,  780,       0,    -760,      0],
    [   0,  150,    0,  1.48e5,       0,  -8040],
    [-1640,   0, -760,       0,  1.52e5,      0],
    [   0,  580,    0,   -8040,       0, 1.53e5],
], dtype=np.float64)

# Maximum misalignment used for energy normalisation (Eq. 11)
MAX_MISALIGN_TRANS_MM = 10.0
MAX_MISALIGN_ROT_RAD  = np.deg2rad(10.0)


# ---------------------------------------------------------------------------
# CT HU VOLUME  —  loader + coordinate helper
# ---------------------------------------------------------------------------

class CTHUVolume:
    """
    CT volume in Hounsfield units held as a GPU tensor.

    Attributes
    ----------
    tensor    : (D, H, W) float32 CUDA tensor
    origin    : (3,) float64 numpy  — physical origin in mm  (x, y, z)
    spacing   : (3,) float64 numpy  — voxel spacing in mm    (sx, sy, sz)
    direction : (3, 3) float64 numpy
    """

    def __init__(self, nrrd_path: str, device: str = 'cuda'):
        img  = sitk.ReadImage(nrrd_path)
        arr  = sitk.GetArrayFromImage(img)                          # (D, H, W)
        self.tensor    = torch.from_numpy(arr.astype(np.float32)).to(device)
        self.origin    = np.array(img.GetOrigin(),    dtype=np.float64)
        self.spacing   = np.array(img.GetSpacing(),   dtype=np.float64)
        self.direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
        self.device    = device

        # Precompute world → voxel matrix
        sp_inv         = np.diag(1.0 / self.spacing)
        self._w2v      = torch.from_numpy(
            (sp_inv @ self.direction.T).astype(np.float32)
        ).to(device)
        self._origin_t = torch.from_numpy(
            self.origin.astype(np.float32)
        ).to(device)

    def world_to_voxel(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (N, 3) world mm  →  (N, 3) continuous voxel coords"""
        return (pts - self._origin_t) @ self._w2v.T

    @property
    def shape_dhw(self):
        return self.tensor.shape   # (D, H, W)


def preload_ct_hu_volumes(cases_dir: str, case_names: list,
                          device: str = 'cuda') -> list:
    """
    Load CT.nrrd (HU values) for every case.
    Returns list of CTHUVolume in the same order as case_names.
    """
    volumes = []
    for case in case_names:
        path = os.path.join(cases_dir, case, 'CT.nrrd')
        if not os.path.exists(path):
            raise FileNotFoundError(f"CT HU volume not found: {path}")
        vol = CTHUVolume(path, device=device)
        volumes.append(vol)
        print(f"  Loaded CT HU for {case}: {vol.shape_dhw}")
    return volumes


# ---------------------------------------------------------------------------
# STEP 1  —  resample transformed CT onto US grid
# ---------------------------------------------------------------------------

def transform_ct_to_us_space(
    ct_hu:              CTHUVolume,
    sitk_transform_inv,               # SimpleITK Euler3DTransform (already inverted)
    us_shape_dhw:       tuple,
    us_origin:          np.ndarray,
    us_spacing:         np.ndarray,
    us_direction:       np.ndarray,
    device:             str = 'cuda',
) -> torch.Tensor:
    """
    For every voxel in the US grid compute the corresponding HU value from
    the (transformed) CT volume via trilinear interpolation.

    Returns
    -------
    ct_in_us : (D_us, H_us, W_us) float32 — HU values on US grid.
               Voxels outside CT FOV filled with -1000 HU (air).
    """
    D, H, W = us_shape_dhw

    # US voxel index grid  →  physical world coordinates
    iz, iy, ix = torch.meshgrid(
        torch.arange(D, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    idx = torch.stack([ix.reshape(-1),
                       iy.reshape(-1),
                       iz.reshape(-1)], dim=1)   # (N,3)  x,y,z

    us_dir_t  = torch.from_numpy(us_direction.astype(np.float32)).to(device)
    us_sp_t   = torch.from_numpy(us_spacing.astype(np.float32)).to(device)
    us_orig_t = torch.from_numpy(us_origin.astype(np.float32)).to(device)
    world_pts = us_orig_t + (idx * us_sp_t) @ us_dir_t.T   # (N,3)

    # US world  →  CT world via fully vectorised affine matrix
    M_ct = sitk_euler_to_matrix(sitk_transform_inv)   # (4,4)
    M_t  = torch.from_numpy(M_ct.astype(np.float32)).to(device)
    N    = world_pts.shape[0]
    pts_h    = torch.cat([world_pts,
                          torch.ones((N, 1), device=device)], dim=1)  # (N,4)
    ct_world = (pts_h @ M_t.T)[:, :3]                                 # (N,3)

    # CT world  →  CT voxel (continuous)
    ct_vox = ct_hu.world_to_voxel(ct_world)   # (N,3)

    # Normalise to [-1,1] for grid_sample  (expects x→W, y→H, z→D)
    Dct, Hct, Wct = ct_hu.shape_dhw
    gx = 2.0 * ct_vox[:, 0] / (Wct - 1) - 1.0
    gy = 2.0 * ct_vox[:, 1] / (Hct - 1) - 1.0
    gz = 2.0 * ct_vox[:, 2] / (Dct - 1) - 1.0
    grid = torch.stack([gx, gy, gz], dim=1).reshape(1, D, H, W, 3)

    ct_vol = ct_hu.tensor.unsqueeze(0).unsqueeze(0)   # (1,1,Dct,Hct,Wct)
    ct_resampled = F.grid_sample(
        ct_vol, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    ).squeeze()   # (D,H,W)

    # Mask voxels that fell outside the CT FOV
    outside = (
        (ct_vox[:, 0] < 0) | (ct_vox[:, 0] > Wct - 1) |
        (ct_vox[:, 1] < 0) | (ct_vox[:, 1] > Hct - 1) |
        (ct_vox[:, 2] < 0) | (ct_vox[:, 2] > Dct - 1)
    ).reshape(D, H, W)
    ct_resampled[outside] = -1000.0

    return ct_resampled   # (D_us, H_us, W_us)


# ---------------------------------------------------------------------------
# STEP 2  —  ray-casting US simulation  (Eqs. 1–7)
# ---------------------------------------------------------------------------

def simulate_us_from_ct(
    ct_hu_us_space: torch.Tensor,   # (D, H, W)  HU on US grid
    us_volume:      torch.Tensor,   # (D, H, W)  real US — used for weight solve
    beam_axis:      int   = 1,
    tau:            float = HU_FULL_REFLECTION_THRESHOLD,
    alpha:          float = LOG_COMPRESS_ALPHA,
) -> torch.Tensor:
    """
    Simulate US from CT (Eqs. 1–7).

    beam_axis:
        0 = SI  1 = AP (default, posterior probe)  2 = LM

    Returns
    -------
    sim_us : (D, H, W) float32
    """
    device = ct_hu_us_space.device
    D, H, W = ct_hu_us_space.shape
    mu = ct_hu_us_space.float()
    U  = us_volume.float()

    # Gradient along beam axis
    if beam_axis == 0:
        grad_mu = torch.diff(mu, dim=0, prepend=mu[:1])
    elif beam_axis == 1:
        grad_mu = torch.diff(mu, dim=1, prepend=mu[:, :1])
    else:
        grad_mu = torch.diff(mu, dim=2, prepend=mu[:, :, :1])

    abs_grad = torch.abs(grad_mu)
    mu_safe  = torch.where(torch.abs(mu) < 1e-3,
                           torch.full_like(mu, 1e-3), mu)

    # Accumulators
    reflection_vol = torch.zeros_like(mu)
    occlusion_vol  = torch.zeros(D, H, W, dtype=torch.bool, device=device)
    bone_iface_vol = torch.zeros(D, H, W, dtype=torch.bool, device=device)

    shape_1 = [D, H, W]
    shape_1[beam_axis] = 1
    I            = torch.ones(shape_1, device=device)
    occluded_acc = torch.zeros(shape_1, dtype=torch.bool, device=device)

    for step in range(mu.shape[beam_axis]):
        sl = [slice(None), slice(None), slice(None)]
        sl[beam_axis] = step
        sl = tuple(sl)

        abs_g = abs_grad[sl].unsqueeze(beam_axis)
        mu_s  = torch.abs(mu_safe[sl]).unsqueeze(beam_axis)

        # Eq. 1  Δr
        dr = torch.clamp(abs_g / (2.0 * mu_s), 0.0, 1.0)

        # Eq. 2  Δt
        dt = torch.clamp(1.0 - (abs_g / (2.0 * mu_s)) ** 2, 0.0, 1.0)

        # Eq. 3
        r_sl = I * dr
        reflection_vol[sl] = r_sl.squeeze(beam_axis)

        # Eq. 4 modification — full reflection at bone edge
        full_refl = abs_g >= tau
        bone_iface_vol[sl] = full_refl.squeeze(beam_axis)

        # Update beam: zero where full reflection occurs
        I = torch.where(full_refl, torch.zeros_like(I), I * dt)

        # Propagate occlusion
        occluded_acc = occluded_acc | (I <= 0.0)
        occlusion_vol[sl] = occluded_acc.squeeze(beam_axis)

    # Eq. 5  Log-compression
    r_log = torch.log1p(alpha * reflection_vol) / np.log(1.0 + alpha)

    # Eq. 6  CT → US intensity mapping + normalise to [0,1]
    p = CT_TO_US_SCALE * mu + CT_TO_US_OFFSET
    p = torch.clamp(p, -500.0, 1500.0)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)

    # Eq. 7  Least-squares weights α, β, γ
    non_occ = ~occlusion_vol

    occ_us = U[occlusion_vol]
    psi    = occ_us.mean() if occ_us.numel() > 0 else torch.tensor(0.0, device=device)

    p_flat = p[non_occ]
    r_flat = r_log[non_occ]
    U_flat = U[non_occ]

    if p_flat.numel() > 3:
        A      = torch.stack([p_flat, r_flat, torch.ones_like(p_flat)], dim=1)
        result = torch.linalg.lstsq(A, U_flat.unsqueeze(1))
        w      = result.solution.squeeze(1)
        aw, bw, gw = w[0], w[1], w[2]
    else:
        aw = bw = torch.tensor(1.0, device=device)
        gw = torch.tensor(0.0, device=device)

    sim_us = torch.zeros_like(U)
    sim_us[non_occ]        = aw * p[non_occ] + bw * r_log[non_occ] + gw
    sim_us[occlusion_vol]  = psi
    sim_us[bone_iface_vol] = U.max()

    return sim_us


# ---------------------------------------------------------------------------
# STEP 3  —  LC2 similarity metric  (Eq. 8)
# ---------------------------------------------------------------------------

def lc2_metric(
    us_real: torch.Tensor,   # (D, H, W)
    us_sim:  torch.Tensor,   # (D, H, W)
    mask:    torch.Tensor,   # (D, H, W) bool
) -> float:
    """
    LC2 = Σ(U - f)² / (N × Var(U))   evaluated inside mask.
    Lower = better match.
    """
    U = us_real.float()[mask]
    f = us_sim.float()[mask]
    N = U.numel()

    if N < 2:
        return 1.0

    var_U = torch.var(U)
    if var_U < 1e-8:
        return 1.0

    return float(((U - f) ** 2).sum() / (N * var_U))


# ---------------------------------------------------------------------------
# STEP 4  —  Biomechanical energy  (Eqs. 9–11)
# ---------------------------------------------------------------------------

def biomechanical_energy(
    transforms_list: list,   # list of SimpleITK Euler3DTransform (inverted)
    K:               int,
) -> float:
    """
    Normalised system energy across all adjacent vertebra pairs.

    x for each pair = [Tx Ty Tz Rx Ry Rz] relative transform in mm / radians.
    U_pair = 0.5 * x^T K x   (Eq. 10)
    E      = sum(U_pair) / (n_pairs * U_max)   (Eq. 11)
    """
    x_max = np.array([
        MAX_MISALIGN_TRANS_MM, MAX_MISALIGN_TRANS_MM, MAX_MISALIGN_TRANS_MM,
        MAX_MISALIGN_ROT_RAD,  MAX_MISALIGN_ROT_RAD,  MAX_MISALIGN_ROT_RAD,
    ])
    U_max = 0.5 * float(x_max @ STIFFNESS_MATRIX @ x_max)

    total_U = 0.0
    n_pairs = 0

    for k in range(K - 1):
        # SimpleITK Euler3DTransform parameters: [rx, ry, rz, tx, ty, tz]
        p_i = np.array(transforms_list[k].GetParameters())
        p_j = np.array(transforms_list[k + 1].GetParameters())

        rel_rot   = p_j[0:3] - p_i[0:3]   # radians
        rel_trans = p_j[3:6] - p_i[3:6]   # mm

        # Reorder to match stiffness matrix: [Tx Ty Tz Rx Ry Rz]
        x = np.concatenate([rel_trans, rel_rot])

        total_U += 0.5 * float(x @ STIFFNESS_MATRIX @ x)
        n_pairs += 1

    if n_pairs == 0 or U_max < 1e-12:
        return 0.0

    return total_U / (n_pairs * U_max)


# ---------------------------------------------------------------------------
# MAIN EVALUATION  —  drop-in replacement for evaluate_group_gpu
# ---------------------------------------------------------------------------

def evaluate_group_gpu_gill(
    flat_params,
    K,
    centers,
    ct_hu_parsers,        # list[CTHUVolume]
    fixed_parser,         # PyNrrdParser — preprocessed US volume
    case_centroids,       # list of (3,) numpy arrays  (kept for API compatibility)
    case_names,
    iteration,
    max_iter,
    sigma:     float = 1.0,   # biomechanical weight σ  (Eq. 12)
    beam_axis: int   = 1,     # AP axis in DHW layout
    device:    str   = 'cuda',
    profile:   bool  = False,
):
    """
    BCLC2 = LC2 - sigma * E   (Eq. 12)

    CMA-ES minimises, so:
        - LC2  is lower-is-better  → minimise directly
        - E    is lower = less realistic pose  → subtract sigma*E to
          discourage unrealistic configurations

    Returns the same 6-tuple your existing logging expects:
    (total_loss, lc2_val, axes_penalty=0, ivd_loss=0, facet_loss=0, ivd_metrics={})
    """

    us_tensor = fixed_parser.get_tensor(False).to(device).float()
    us_shape  = us_tensor.shape

    us_img       = sitk.ReadImage(fixed_parser.file_path)
    us_origin    = np.array(us_img.GetOrigin(),    dtype=np.float64)
    us_spacing   = np.array(us_img.GetSpacing(),   dtype=np.float64)
    us_direction = np.array(us_img.GetDirection(), dtype=np.float64).reshape(3, 3)

    total_lc2       = 0.0
    num_valid       = 0
    transforms_list = []

    for k in range(K):
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k : 6*(k+1)].tolist())
        tx_inv = tx.GetInverse()
        transforms_list.append(tx_inv)

        # Resample CT HU onto US grid
        ct_in_us = transform_ct_to_us_space(
            ct_hu              = ct_hu_parsers[k],
            sitk_transform_inv = tx_inv,
            us_shape_dhw       = us_shape,
            us_origin          = us_origin,
            us_spacing         = us_spacing,
            us_direction       = us_direction,
            device             = device,
        )

        # Simulate US from CT  (Eqs. 1–7)
        sim_us = simulate_us_from_ct(
            ct_hu_us_space = ct_in_us,
            us_volume      = us_tensor,
            beam_axis      = beam_axis,
        )

        # LC2 over region where CT has content
        valid_mask = ct_in_us > -900.0
        if valid_mask.sum() > 10:
            total_lc2 += lc2_metric(us_tensor, sim_us, valid_mask)
            num_valid += 1

    mean_lc2 = total_lc2 / max(num_valid, 1)

    # Biomechanical energy E  (Eqs. 9–11)
    E = biomechanical_energy(transforms_list, K)

    # Eq. 12
    total_loss = mean_lc2 - sigma * E

    return (
        total_loss,
        mean_lc2,
        0.0,   # axes_penalty  — not in Gill
        0.0,   # ivd_loss      — not in Gill
        0.0,   # facet_loss    — not in Gill
        {},
    )