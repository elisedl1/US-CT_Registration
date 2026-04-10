import os
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

# ── File paths ────────────────────────────────────────────────────────────────
US_file   = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/US_complete_cal.nrrd"
US_mask   = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/US_complete_cal_mask.nrrd"
transform = sitk.ReadTransform('/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/acq0_modif-ImageToReference.h5')

out_dir  = os.path.dirname(US_file)
out_path = os.path.join(out_dir, "US_forward_ray_tracing.nrrd")

# ── Probe geometry ────────────────────────────────────────────────────────────
R     = np.array(transform.GetMatrix()).reshape(3, 3)
z_us  = np.array([0, 0, 1])   # beam axis in probe space
z_ref = R @ z_us               # beam direction in world space (shallow → deep)

# ── Load images ───────────────────────────────────────────────────────────────
US_sitk      = sitk.ReadImage(US_file)
US_mask_sitk = sitk.ReadImage(US_mask)

dir_matrix = np.array(US_sitk.GetDirection()).reshape(3, 3)
spacing    = np.array(US_sitk.GetSpacing())

def phys_to_voxel(phys_vec):
    return np.linalg.inv(dir_matrix) @ (phys_vec / spacing)

z_voxel      = phys_to_voxel(z_ref)
z_voxel_norm = z_voxel / np.linalg.norm(z_voxel)

US_np      = sitk.GetArrayFromImage(US_sitk).transpose(2, 1, 0).astype(np.float32)
US_mask_np = sitk.GetArrayFromImage(US_mask_sitk).transpose(2, 1, 0).astype(np.float32)

intensity_threshold = 0.80 * US_np.max()
print(f"Intensity threshold: {intensity_threshold:.4f}  (80% of max {US_np.max():.4f})")

nx, ny, nz = US_np.shape

# ── Depth map along beam direction ────────────────────────────────────────────
xx, yy, zz = np.meshgrid(
    np.arange(nx, dtype=np.float32),
    np.arange(ny, dtype=np.float32),
    np.arange(nz, dtype=np.float32),
    indexing="ij",
)

beam = z_voxel_norm
depth_map = (xx * beam[0] + yy * beam[1] + zz * beam[2])

mm_per_depth_unit = np.linalg.norm(beam * spacing)
margin_mm         = 8.0
margin_voxels     = margin_mm / mm_per_depth_unit
print(f"mm per depth unit: {mm_per_depth_unit:.4f}  →  8 mm margin = {margin_voxels:.1f} depth units")

# ── Scan-line grouping (perpendicular to beam) ────────────────────────────────
if abs(beam[0]) < 0.9:
    tmp = np.array([1, 0, 0])
else:
    tmp = np.array([0, 1, 0])
lat1 = np.cross(beam, tmp); lat1 /= np.linalg.norm(lat1)
lat2 = np.cross(beam, lat1); lat2 /= np.linalg.norm(lat2)

coords_flat = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
l1     = coords_flat @ lat1
l2     = coords_flat @ lat2
d      = coords_flat @ beam

l1_idx = np.round(l1).astype(np.int32)
l2_idx = np.round(l2).astype(np.int32)

US_flat   = US_np.ravel()
mask_flat = US_mask_np.ravel()
output_np = US_np.copy()
out_flat  = output_np.ravel()

print("Building scan-line index …")
scanline_dict = defaultdict(list)
for flat_i, key in enumerate(zip(l1_idx.tolist(), l2_idx.tolist())):
    scanline_dict[key].append(flat_i)

print(f"Number of scan lines: {len(scanline_dict)}")

# ── Forward ray-tracing ───────────────────────────────────────────────────────
# Traverse shallow → deep (ascending depth order).
# First masked voxel >= threshold = bone surface.
# Keep window: [bone_depth, bone_depth + 8mm].
# Zero out everything outside that window.

print("Tracing rays …")
n_processed = 0
for key, indices in scanline_dict.items():
    indices  = np.array(indices)
    depths   = d[indices]
    intens   = US_flat[indices]
    in_mask  = mask_flat[indices] > 0

    # Sort shallow → deep (ascending)
    order     = np.argsort(depths)
    indices_s = indices[order]
    depths_s  = depths[order]
    intens_s  = intens[order]
    in_mask_s = in_mask[order]

    # Find first masked voxel above threshold coming from the probe side
    bone_depth = None
    for j in range(len(indices_s)):
        if in_mask_s[j] and intens_s[j] >= intensity_threshold:
            bone_depth = depths_s[j]
            break

    if bone_depth is None:
        # No bone found – zero out entire scan line inside mask
        out_flat[indices_s[in_mask_s]] = 0.0
        continue

    # Keep window: [bone_depth, bone_depth + 8mm] (bone surface + 8mm deeper)
    # Zero out everything outside this window
    keep_min = bone_depth                   # bone surface (shallowest kept point)
    keep_max = bone_depth + margin_voxels   # 8 mm deeper into bone/shadow

    for j in range(len(indices_s)):
        if in_mask_s[j]:
            if depths_s[j] < keep_min or depths_s[j] > keep_max:
                out_flat[indices_s[j]] = 0.0

    n_processed += 1

print(f"Scan lines with detected bone surface: {n_processed}")

# ── Save result ───────────────────────────────────────────────────────────────
def save_as_nrrd(array_xyz, reference_sitk, path):
    arr_zyx  = array_xyz.transpose(2, 1, 0)
    out_sitk = sitk.GetImageFromArray(arr_zyx)
    out_sitk.SetSpacing(reference_sitk.GetSpacing())
    out_sitk.SetOrigin(reference_sitk.GetOrigin())
    out_sitk.SetDirection(reference_sitk.GetDirection())
    sitk.WriteImage(out_sitk, path)
    print(f"Saved: {path}")

save_as_nrrd(output_np, US_sitk, out_path)
print("Done.")