"""
Standalone debug script for US simulation.
Loads a real US volume and a CT volume, runs the simulation pipeline,
and saves slice visualisations at each stage so you can inspect what's happening.
"""

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# --- PATHS — edit these ---
US_PATH  = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa1/Cases/US_complete_cal.nrrd"
CT_PATH  = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa1/Cases/L3/CT.nrrd"   # pick any one vertebra
OUT_DIR  = "./debug_simulation_output"
os.makedirs(OUT_DIR, exist_ok=True)

BEAM_AXIS = 1   # 0=SI, 1=AP, 2=LM  — change if needed

# ---------------------------------------------------------------------------
# Inline simulation (no import needed — self contained)
# ---------------------------------------------------------------------------
HU_FULL_REFLECTION_THRESHOLD = 250.0
LOG_COMPRESS_ALPHA           = 10.0
CT_TO_US_SCALE               = 1.36
CT_TO_US_OFFSET              = -1429.0


def load_volume(path, device='cpu'):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (D, H, W)
    t   = torch.from_numpy(arr).to(device)
    print(f"  Loaded {os.path.basename(path)}: shape={arr.shape}  "
          f"range=[{arr.min():.1f}, {arr.max():.1f}]  "
          f"spacing={img.GetSpacing()}")
    return t, img


def simulate_us_from_ct(ct_hu_us_space, us_volume, beam_axis=1,
                        tau=HU_FULL_REFLECTION_THRESHOLD,
                        alpha=LOG_COMPRESS_ALPHA):
    device = ct_hu_us_space.device
    D, H, W = ct_hu_us_space.shape
    mu = ct_hu_us_space.float()
    U  = us_volume.float()

    # gradient along beam axis
    if beam_axis == 0:
        grad_mu = torch.diff(mu, dim=0, prepend=mu[:1])
    elif beam_axis == 1:
        grad_mu = torch.diff(mu, dim=1, prepend=mu[:, :1])
    else:
        grad_mu = torch.diff(mu, dim=2, prepend=mu[:, :, :1])

    abs_grad = torch.abs(grad_mu)
    mu_safe  = torch.where(torch.abs(mu) < 1e-3,
                           torch.full_like(mu, 1e-3), mu)

    print(f"\n  mu range:       {mu.min():.1f}  to  {mu.max():.1f}")
    print(f"  abs_grad range: {abs_grad.min():.4f}  to  {abs_grad.max():.4f}")
    print(f"  mu_safe ~zero:  {(torch.abs(mu_safe) < 1e-2).sum().item()} voxels")

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

        dr = torch.clamp(abs_g / (2.0 * mu_s), 0.0, 1.0)
        dt = torch.clamp(1.0 - (abs_g / (2.0 * mu_s)) ** 2, 0.0, 1.0)

        r_sl = I * dr
        reflection_vol[sl] = r_sl.squeeze(beam_axis)

        full_refl = abs_g >= tau
        bone_iface_vol[sl] = full_refl.squeeze(beam_axis)

        I = torch.where(full_refl, torch.zeros_like(I), I * dt)
        occluded_acc = occluded_acc | (I <= 0.0)
        occlusion_vol[sl] = occluded_acc.squeeze(beam_axis)

    print(f"  bone interfaces: {bone_iface_vol.sum().item()} voxels")
    print(f"  occluded:        {occlusion_vol.sum().item()} voxels "
          f"({100*occlusion_vol.float().mean().item():.1f}%)")

    # log compress
    r_log = torch.log1p(alpha * reflection_vol) / np.log(1.0 + alpha)
    print(f"  r_log NaN: {torch.isnan(r_log).sum().item()}")

    # CT → US mapping
    p = CT_TO_US_SCALE * mu + CT_TO_US_OFFSET
    p = torch.clamp(p, -500.0, 1500.0)
    p_min, p_max = p.min(), p.max()
    print(f"  p range before norm: {p_min:.1f} to {p_max:.1f}")
    p = (p - p_min) / (p_max - p_min + 1e-8)
    print(f"  p NaN: {torch.isnan(p).sum().item()}")

    # least-squares weights
    non_occ = ~occlusion_vol
    occ_us  = U[occlusion_vol]
    psi     = occ_us.mean() if occ_us.numel() > 0 else torch.tensor(0.0)

    p_flat = p[non_occ]
    r_flat = r_log[non_occ]
    U_flat = U[non_occ]

    print(f"  non-occluded voxels for lstsq: {p_flat.numel()}")

    if p_flat.numel() > 3:
        A      = torch.stack([p_flat, r_flat, torch.ones_like(p_flat)], dim=1)
        result = torch.linalg.lstsq(A, U_flat.unsqueeze(1))
        w      = result.solution.squeeze(1)
        aw, bw, gw = w[0], w[1], w[2]
        print(f"  lstsq weights: aw={aw:.4f}  bw={bw:.4f}  gw={gw:.4f}")
    else:
        print("  WARNING: too few non-occluded voxels — using fallback weights")
        aw = bw = torch.tensor(1.0)
        gw = torch.tensor(0.0)

    sim_us = torch.zeros_like(U)
    sim_us[non_occ]        = aw * p[non_occ] + bw * r_log[non_occ] + gw
    sim_us[occlusion_vol]  = psi
    sim_us[bone_iface_vol] = U.max()

    print(f"  sim_us NaN: {torch.isnan(sim_us).sum().item()}")
    print(f"  sim_us range: {sim_us.min():.4f} to {sim_us.max():.4f}")

    return sim_us, r_log, p, occlusion_vol, bone_iface_vol


def save_slice_comparison(volumes_dict, axis, slice_idx, out_path):
    """
    Save side-by-side slices from multiple volumes along a given axis.
    volumes_dict: {label: tensor (D,H,W)}
    """
    n = len(volumes_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, vol) in zip(axes, volumes_dict.items()):
        arr = vol.cpu().numpy()
        if axis == 0:
            sl = arr[slice_idx]
        elif axis == 1:
            sl = arr[:, slice_idx, :]
        else:
            sl = arr[:, :, slice_idx]

        vmin, vmax = np.percentile(arr[arr != 0], [2, 98]) if (arr != 0).any() else (0, 1)
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"{label}\nslice {slice_idx} axis {axis}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN DEBUG
# ---------------------------------------------------------------------------
print("="*60)
print("Loading volumes...")
print("="*60)

us_tensor, us_img = load_volume(US_PATH)
ct_tensor, ct_img = load_volume(CT_PATH)

# The CT is likely a different size/spacing than US.
# For this debug we just use the CT as-is (no registration transform)
# to check the simulation pipeline itself.
# We'll crop/pad to match US shape for a rough check.
D_us, H_us, W_us = us_tensor.shape
D_ct, H_ct, W_ct = ct_tensor.shape

print(f"\nUS shape:  {us_tensor.shape}")
print(f"CT shape:  {ct_tensor.shape}")

# For simulation debug: resample CT to US shape using grid_sample
print("\nResampling CT to US shape for debug...")
ct_vol = ct_tensor.unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)

# identity grid in US space
iz, iy, ix = torch.meshgrid(
    torch.linspace(-1, 1, D_us),
    torch.linspace(-1, 1, H_us),
    torch.linspace(-1, 1, W_us),
    indexing='ij'
)
grid = torch.stack([ix, iy, iz], dim=-1).unsqueeze(0)   # (1,D,H,W,3)

import torch.nn.functional as F
ct_resampled = F.grid_sample(
    ct_vol, grid, mode='bilinear', padding_mode='border', align_corners=True
).squeeze()   # (D_us, H_us, W_us)

print(f"Resampled CT shape: {ct_resampled.shape}  "
      f"range=[{ct_resampled.min():.1f}, {ct_resampled.max():.1f}]")

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"Running US simulation (beam_axis={BEAM_AXIS})...")
print("="*60)

sim_us, r_log, p_mapped, occlusion, bone_iface = simulate_us_from_ct(
    ct_resampled, us_tensor, beam_axis=BEAM_AXIS
)

# ---------------------------------------------------------------------------
# Save visualisations at three representative slices
# ---------------------------------------------------------------------------
print("\nSaving slice visualisations...")

for axis in [0, 1, 2]:
    size_along_axis = [D_us, H_us, W_us][axis]
    mid = size_along_axis // 2

    save_slice_comparison(
        {
            "US (real)":        us_tensor,
            "CT (resampled HU)": ct_resampled,
            "p (CT→US mapped)": p_mapped,
            "r_log (reflections)": r_log,
            "sim_US":           sim_us,
        },
        axis=axis,
        slice_idx=mid,
        out_path=os.path.join(OUT_DIR, f"debug_axis{axis}_slice{mid}.png")
    )

# also save occlusion and bone interface masks
for axis in [0, 1, 2]:
    size_along_axis = [D_us, H_us, W_us][axis]
    mid = size_along_axis // 2

    save_slice_comparison(
        {
            "occlusion mask":    occlusion.float(),
            "bone interface":    bone_iface.float(),
        },
        axis=axis,
        slice_idx=mid,
        out_path=os.path.join(OUT_DIR, f"debug_masks_axis{axis}_slice{mid}.png")
    )

# ---------------------------------------------------------------------------
# LC2 check
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("LC2 check...")
print("="*60)

valid = ct_resampled > -900.0
U_v   = us_tensor[valid].float()
f_v   = sim_us[valid].float()
N     = U_v.numel()
var_U = torch.var(U_v)

print(f"  Valid voxels: {N}")
print(f"  Var(U):       {var_U:.6f}")

if var_U > 1e-8 and N > 1:
    lc2 = float(((U_v - f_v)**2).sum() / (N * var_U))
    print(f"  LC2:          {lc2:.6f}")
else:
    print("  LC2:          undefined (zero variance or too few voxels)")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")