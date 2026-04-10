import os
import numpy as np
import torch
import SimpleITK as sitk

# ── File paths ────────────────────────────────────────────────────────────────
US_file   = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/US_complete_cal.nrrd"
US_mask   = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/US_complete_cal_mask.nrrd"
transform = sitk.ReadTransform('/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/acq0_modif-ImageToReference.h5')

out_dir   = os.path.dirname(US_file)
out_path  = os.path.join(out_dir, "US_ray_tracing.nrrd")

# ── Probe geometry (same as DGC script) ──────────────────────────────────────
R = np.array(transform.GetMatrix()).reshape(3, 3)

z_us  = np.array([0, 0, 1])          # beam / depth axis in probe space
z_ref = R @ z_us                      # beam direction in reference (world) space

# ── Load images ───────────────────────────────────────────────────────────────
US_sitk      = sitk.ReadImage(US_file)
US_mask_sitk = sitk.ReadImage(US_mask)

dir_matrix = np.array(US_sitk.GetDirection()).reshape(3, 3)
spacing    = np.array(US_sitk.GetSpacing())          # (sx, sy, sz) in mm

# world → voxel direction (no translation needed for direction vectors)
def phys_to_voxel(phys_vec):
    return np.linalg.inv(dir_matrix) @ (phys_vec / spacing)

z_voxel      = phys_to_voxel(z_ref)
z_voxel_norm = z_voxel / np.linalg.norm(z_voxel)    # unit vector along beam

# Arrays in X × Y × Z order (same as DGC script)
US_np      = sitk.GetArrayFromImage(US_sitk).transpose(2, 1, 0).astype(np.float32)
US_mask_np = sitk.GetArrayFromImage(US_mask_sitk).transpose(2, 1, 0).astype(np.float32)

# Image is assumed already normalised [0, 1] as stated by the user
intensity_threshold = 0.20 * US_np.max()
print(f"Intensity threshold: {intensity_threshold:.4f}  (20 % of max {US_np.max():.4f})")

nx, ny, nz = US_np.shape

# ── Depth map along beam direction ────────────────────────────────────────────
# depth_map[x,y,z] = projection of voxel coordinate onto the beam axis
xx, yy, zz = np.meshgrid(
    np.arange(nx, dtype=np.float32),
    np.arange(ny, dtype=np.float32),
    np.arange(nz, dtype=np.float32),
    indexing="ij",
)
depth_map = (xx * z_voxel_norm[0] +
             yy * z_voxel_norm[1] +
             zz * z_voxel_norm[2])   # scalar projection (voxel units along beam)

# Physical depth in mm at each voxel:
# one voxel step along z_voxel_norm corresponds to |spacing * z_voxel_norm| mm
# but since z_voxel_norm was derived from phys_to_voxel (which divides by spacing),
# depth_map is in *isotropic* normalised voxel units.  Convert to mm:
# The physical length of z_voxel before normalisation is ||phys_to_voxel(z_ref)||,
# which represents the voxel-space length per unit world length.
# Simpler: voxel depth in mm = depth_map_voxels / norm(phys_to_voxel(z_ref_unit)) * 1 mm
# We just need the mm-per-voxel-depth-unit:
voxel_depth_to_mm = 1.0 / np.linalg.norm(phys_to_voxel(z_ref / np.linalg.norm(z_ref)))
# Actually the simplest: along the normalised voxel direction z_voxel_norm,
# moving 1 voxel-index unit corresponds to this many mm:
mm_per_depth_unit = np.linalg.norm(z_voxel_norm * spacing)   # mm

margin_mm      = 8.0                                          # keep 8 mm above bone
margin_voxels  = margin_mm / mm_per_depth_unit
print(f"mm per depth unit: {mm_per_depth_unit:.4f}  →  8 mm margin = {margin_voxels:.1f} depth units")

# ── Backward ray-tracing ──────────────────────────────────────────────────────
# The DGC filter increases weight *with* depth (deeper = more weight).
# Backward ray-tracing goes in the OPPOSITE direction:
#   start from the deepest voxel on each scan line, travel toward the probe,
#   stop at the first voxel whose intensity ≥ threshold  (= bone surface).
# Everything deeper than (bone_depth - margin) is set to zero.

output_np = US_np.copy()

# We iterate over planes perpendicular to the beam.
# Strategy: for each voxel, ask "is there a bone voxel at greater depth on the
# same scan line?"  A scan line is a set of voxels sharing the same (x,y,z)
# coordinates *after removing the beam component*, i.e. same lateral position.

# Lateral coordinates (2-D plane perpendicular to the beam):
# Pick two arbitrary perpendicular axes to the beam
beam = z_voxel_norm
if abs(beam[0]) < 0.9:
    tmp = np.array([1, 0, 0])
else:
    tmp = np.array([0, 1, 0])
lat1 = np.cross(beam, tmp); lat1 /= np.linalg.norm(lat1)
lat2 = np.cross(beam, lat1); lat2 /= np.linalg.norm(lat2)

# Project every voxel onto the two lateral axes → (lat1_coord, lat2_coord, depth)
coords_flat = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # (N,3)

l1 = coords_flat @ lat1   # (N,)
l2 = coords_flat @ lat2   # (N,)
d  = coords_flat @ beam   # (N,)  – depth along beam

# Discretise lateral coords to group scan-line neighbours.
# Resolution: 1 voxel in lateral direction
l1_idx = np.round(l1).astype(np.int32)
l2_idx = np.round(l2).astype(np.int32)

# Flat intensity and mask arrays
US_flat   = US_np.ravel()
mask_flat = US_mask_np.ravel()
out_flat  = output_np.ravel()   # view into output_np

# Group voxels by scan-line key
from collections import defaultdict
scanline_dict = defaultdict(list)
N = len(US_flat)
idx_all = np.arange(N)

print("Building scan-line index …")
keys = list(zip(l1_idx.tolist(), l2_idx.tolist()))
for flat_i, key in enumerate(keys):
    scanline_dict[key].append(flat_i)

print(f"Number of scan lines: {len(scanline_dict)}")

# Process each scan line
print("Tracing rays …")
n_processed = 0
for (key, indices) in scanline_dict.items():
    indices  = np.array(indices)
    depths   = d[indices]
    intens   = US_flat[indices]
    in_mask  = mask_flat[indices] > 0

    # Sort by depth descending (deepest first = backward from shadow)
    order       = np.argsort(depths)[::-1]
    indices_s   = indices[order]
    depths_s    = depths[order]
    intens_s    = intens[order]
    in_mask_s   = in_mask[order]

    # Find the first (deepest) voxel inside the mask that exceeds threshold
    bone_depth = None
    for j in range(len(indices_s)):
        if in_mask_s[j] and intens_s[j] >= intensity_threshold:
            bone_depth = depths_s[j]
            break

    if bone_depth is None:
        # No bone found – zero out entire scan line inside mask
        out_flat[indices_s[in_mask_s]] = 0.0
        continue

    # Keep only the 8 mm window from the bone surface back toward the probe:
    #   keep_max = bone_depth          (the surface itself, deepest kept point)
    #   keep_min = bone_depth - margin (8 mm shallower, toward probe)
    # Everything outside this window is zeroed out.
    keep_min = bone_depth - margin_voxels   # toward probe
    keep_max = bone_depth                   # bone surface

    for j in range(len(indices_s)):
        if in_mask_s[j]:
            if depths_s[j] < keep_min or depths_s[j] > keep_max:
                out_flat[indices_s[j]] = 0.0

    n_processed += 1

print(f"Scan lines with detected bone surface: {n_processed}")

# ── Save result ───────────────────────────────────────────────────────────────
def save_as_nrrd(array_xyz, reference_sitk, path):
    """Save X×Y×Z numpy array as NRRD, copying geometry from reference."""
    arr_zyx = array_xyz.transpose(2, 1, 0)
    out_sitk = sitk.GetImageFromArray(arr_zyx)
    out_sitk.SetSpacing(reference_sitk.GetSpacing())
    out_sitk.SetOrigin(reference_sitk.GetOrigin())
    out_sitk.SetDirection(reference_sitk.GetDirection())
    sitk.WriteImage(out_sitk, path)
    print(f"Saved: {path}")

save_as_nrrd(output_np, US_sitk, out_path)
print("Done.")