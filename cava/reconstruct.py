#!/usr/bin/env python3
"""
reconstruct_hus.py — 3D Freehand US Reconstruction (CAVA Dataset)
=================================================================

Stacks 2D HUS PNG frames into a 3D NRRD volume using tracked probe poses.

Transform chain (from calibration README):
  1. Pixel (u,v) → p_I = [(u)*S_x, (v)*S_y, 0, 1]     (mm, image frame)
  2. p_MUS = T_MUS_I @ p_I                               (marker frame)
  3. p_C   = T_C_MUS[k] @ p_MUS                          (camera frame, per-frame)

Required inputs:
  --data_dir   : folder containing HUS/ (PNGs) and HUS_pose.txt
  --calib_yaml : US_calibration.yaml with S_x, S_y, T_SI

Deps: pip install numpy scipy Pillow pynrrd pyyaml tqdm
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import nrrd
except ImportError:
    sys.exit("Install pynrrd: pip install pynrrd")
try:
    import yaml
except ImportError:
    sys.exit("Install PyYAML: pip install pyyaml")


# ── Rotation helpers ────────────────────────────────────────────────────────

def quat_to_matrix(qx, qy, qz, qw):
    """Quaternion → 3×3 rotation matrix (Hamilton, scalar-last)."""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def make_4x4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── Load calibration YAML ──────────────────────────────────────────────────

def load_calibration(yaml_path):
    """
    Returns S_x (mm/px), S_y (mm/px), T_MUS_I (4×4 numpy array).
    T_MUS_I = handheld.T_SI from the YAML = image {I} → marker {MUS}.
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    hus = raw['handheld']
    S_x = float(hus['S_x'])
    S_y = float(hus['S_y'])

    # T_SI could be: a flat list of 16 floats, a 4×4 nested list, or a numpy-like string
    t_raw = hus['T_SI']
    if isinstance(t_raw, list):
        flat = np.array(t_raw, dtype=float).flatten()
        if flat.size == 16:
            T = flat.reshape(4, 4)
        else:
            raise ValueError(f"T_SI has {flat.size} elements, expected 16")
    elif isinstance(t_raw, str):
        # Handle numpy-style string repr
        T = np.array(eval(t_raw), dtype=float).reshape(4, 4)
    else:
        raise ValueError(f"Unexpected T_SI type: {type(t_raw)}")

    print(f"[calib] S_x={S_x:.6f}  S_y={S_y:.6f} mm/px")
    print(f"[calib] T_MUS_I:\n{T}")
    return S_x, S_y, T


# ── Load HUS poses ─────────────────────────────────────────────────────────

def load_poses(pose_path):
    """
    Parse HUS_pose.txt.
    Supports 15-col format: tx ty tz qx qy qz qw  [ref×7]  timestamp
    Also supports 8-col:    tx ty tz qx qy qz qw  timestamp

    Returns:
        transforms : list[np.ndarray]  — T_C_MUS per row (4×4)
        timestamps : np.ndarray        — timestamps
    """
    transforms = []
    timestamps = []

    with open(pose_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha() or line.startswith('#'):
                continue

            vals = [float(x) for x in line.split()]

            if len(vals) >= 15:
                tx, ty, tz = vals[0], vals[1], vals[2]
                qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
                ts = vals[14]
            elif len(vals) >= 8:
                tx, ty, tz = vals[0], vals[1], vals[2]
                qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
                ts = vals[7]
            else:
                continue

            R = quat_to_matrix(qx, qy, qz, qw)
            T = make_4x4(R, np.array([tx, ty, tz]))
            transforms.append(T)
            timestamps.append(ts)

    timestamps = np.array(timestamps)
    print(f"[poses] {len(transforms)} poses loaded")
    print(f"[poses] time: {timestamps[0]:.4f} → {timestamps[-1]:.4f}  "
          f"(Δ = {timestamps[-1]-timestamps[0]:.2f} s)")
    return transforms, timestamps


# ── Find and sort PNG frames ───────────────────────────────────────────────

def find_frames(data_dir):
    """Look for PNGs in data_dir/HUS/ or data_dir/."""
    for subdir in ['HUS', 'XUS', '.']:
        pat = os.path.join(data_dir, subdir, '*.png')
        files = sorted(glob.glob(pat))
        if files:
            print(f"[frames] {len(files)} PNGs in {os.path.dirname(pat) or '.'}")
            return files
    return []


# ── Synchronize frames ↔ poses via timestamps ─────────────────────────────

def sync_frames_to_poses(frame_paths, pose_timestamps, fps=15.0):
    """
    PNGs are named sequentially (000000.png …). Frame timestamps are
    derived as  t = t_first_pose + frame_index / fps.

    Returns list of (frame_idx, pose_idx) tuples.
    """
    n = len(frame_paths)
    # Extract integer indices from filenames
    indices = []
    for p in frame_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            indices.append(int(stem))
        except ValueError:
            indices.append(len(indices))
    indices = np.array(indices, dtype=float)

    t0 = pose_timestamps[0]
    frame_ts = t0 + indices / fps

    matched = []
    for fi in range(n):
        diffs = np.abs(pose_timestamps - frame_ts[fi])
        pi = int(np.argmin(diffs))
        if diffs[pi] < 0.2:          # 200 ms tolerance (~3 frames)
            matched.append((fi, pi))

    print(f"[sync]  {len(matched)} / {n} frames matched to poses")
    return matched


# ── Core: bin-average compounding ──────────────────────────────────────────

def reconstruct(frame_paths, matched, transforms, S_x, S_y, T_MUS_I,
                voxel_mm=0.5, subsample=2):
    """
    Two-pass voxelization:
      Pass 1  — scan all frame corners to find bounding box
      Pass 2  — scatter pixel intensities into voxel grid, average

    Returns volume (3D float32), origin (3,), spacing (3,).
    """

    # ---- Image size from first matched frame ----
    sample = np.array(Image.open(frame_paths[matched[0][0]]).convert('L'))
    H, W = sample.shape
    print(f"[recon] frame {W}×{H} px  →  {W*S_x:.1f}×{H*S_y:.1f} mm")

    # ---- Precompute subsampled pixel grid in image coords (mm) ----
    us = np.arange(0, W, subsample, dtype=np.float64)
    vs = np.arange(0, H, subsample, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)          # shape (nv, nu)
    npx = uu.size

    pix_img = np.ones((npx, 4), dtype=np.float64)
    pix_img[:, 0] = uu.ravel() * S_x     # x in mm
    pix_img[:, 1] = vv.ravel() * S_y     # y in mm
    pix_img[:, 2] = 0.0                  # z = 0 (image plane)

    # Pre-multiply calibration (constant): p_MUS = T_MUS_I @ p_I
    pix_marker = (T_MUS_I @ pix_img.T).T  # (npx, 4) in marker frame

    # Integer indices for sampling the image at subsampled locations
    uu_i = np.clip(uu.ravel().astype(int), 0, W - 1)
    vv_i = np.clip(vv.ravel().astype(int), 0, H - 1)

    # ---- Image corners in marker frame (for bbox) ----
    corners_img = np.array([
        [0,         0,         0, 1],
        [(W-1)*S_x, 0,         0, 1],
        [0,         (H-1)*S_y, 0, 1],
        [(W-1)*S_x, (H-1)*S_y, 0, 1],
    ])
    corners_mk = (T_MUS_I @ corners_img.T).T

    # ---- PASS 1: bounding box ----
    print("[recon] pass 1 — bounding box")
    pts = []
    for fi, pi in tqdm(matched, desc="bbox", leave=False):
        T_C_MUS = transforms[pi]
        c = (T_C_MUS @ corners_mk.T).T[:, :3]
        pts.append(c)
    pts = np.vstack(pts)

    pad = 5.0   # mm margin
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    dims = np.ceil((hi - lo) / voxel_mm).astype(int)
    nvox = int(np.prod(dims))

    print(f"[recon] bbox: {lo} → {hi}")
    print(f"[recon] grid: {dims}  ({nvox/1e6:.1f} M voxels)")
    if nvox > 2_000_000_000:
        sys.exit("[ERROR] Volume too large — increase --voxel_size")

    # ---- PASS 2: scatter ----
    print("[recon] pass 2 — voxelizing")
    vol_sum   = np.zeros(dims, dtype=np.float64)
    vol_count = np.zeros(dims, dtype=np.uint16)
    inv_v = 1.0 / voxel_mm
    d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])

    for fi, pi in tqdm(matched, desc="voxelize"):
        img = np.array(Image.open(frame_paths[fi]).convert('L'))
        intensities = img[vv_i, uu_i].astype(np.float64)

        # skip blank / no-contact frames
        if intensities.mean() < 5.0:
            continue

        T_C_MUS = transforms[pi]
        world = (T_C_MUS @ pix_marker.T).T[:, :3]

        idx = ((world - lo) * inv_v).astype(np.int32)
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < d0) & \
             (idx[:, 1] >= 0) & (idx[:, 1] < d1) & \
             (idx[:, 2] >= 0) & (idx[:, 2] < d2)

        vi = idx[ok]
        iv = intensities[ok]

        flat = vi[:, 0] * (d1 * d2) + vi[:, 1] * d2 + vi[:, 2]
        np.add.at(vol_sum.ravel(),   flat, iv)
        np.add.at(vol_count.ravel(), flat, 1)

    # ---- average + light smooth ----
    mask = vol_count > 0
    vol = np.zeros(dims, dtype=np.float32)
    vol[mask] = (vol_sum[mask] / vol_count[mask]).astype(np.float32)
    vol = gaussian_filter(vol, sigma=0.5).astype(np.float32)

    pct = mask.sum() / max(mask.size, 1) * 100
    print(f"[recon] filled {mask.sum():,} / {mask.size:,} voxels ({pct:.1f}%)")

    spacing = np.array([voxel_mm, voxel_mm, voxel_mm])
    return vol, lo, spacing


# ── Save NRRD ──────────────────────────────────────────────────────────────

def save_nrrd(vol, origin, spacing, path):
    header = {
        'space': 'left-posterior-superior',
        'space directions': np.diag(spacing).tolist(),
        'space origin': origin.tolist(),
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'gzip',
        'type': 'float',
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    nrrd.write(path, vol, header)
    mb = os.path.getsize(path) / 1048576
    print(f"[save]  {path}  ({mb:.1f} MB)")
    print(f"        shape={vol.shape}  spacing={spacing}  origin={origin}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HUS → 3D NRRD (CAVA dataset)")
    ap.add_argument('--data_dir',   required=True,
                    help='Scan folder with HUS/ and HUS_pose.txt')
    ap.add_argument('--calib_yaml', required=True,
                    help='Path to US_calibration.yaml')
    ap.add_argument('--output',     default=None,
                    help='Output .nrrd path')
    ap.add_argument('--voxel_size', type=float, default=0.5,
                    help='Voxel size in mm (default 0.5)')
    ap.add_argument('--subsample',  type=int, default=2,
                    help='Pixel subsampling (default 2)')
    args = ap.parse_args()

    # 1. calibration
    S_x, S_y, T_MUS_I = load_calibration(args.calib_yaml)

    # 2. frames
    frames = find_frames(args.data_dir)
    if not frames:
        sys.exit(f"[ERROR] No PNGs in {args.data_dir}/HUS/")

    # 3. poses
    pose_path = os.path.join(args.data_dir, 'HUS_pose.txt')
    if not os.path.exists(pose_path):
        sys.exit(f"[ERROR] Not found: {pose_path}")
    transforms, timestamps = load_poses(pose_path)

    # 4. sync
    matched = sync_frames_to_poses(frames, timestamps, fps=15.0)
    if not matched:
        sys.exit("[ERROR] No frames matched to poses")

    # 5. reconstruct
    vol, origin, spacing = reconstruct(
        frames, matched, transforms,
        S_x, S_y, T_MUS_I,
        voxel_mm=args.voxel_size,
        subsample=args.subsample,
    )

    # 6. save
    out = args.output or os.path.join(args.data_dir, 'recon.nrrd')
    save_nrrd(vol, origin, spacing, out)
    print("[done]")


if __name__ == '__main__':
    main()