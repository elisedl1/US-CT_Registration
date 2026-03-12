#!/usr/bin/env python3
"""
reconstruct_hus.py — 3D Freehand US Reconstruction (CAVA Dataset)
=================================================================

The frame-grabber captures 1920×1080 PNGs that include the Aixplorer UI.
Only a rectangular sub-region contains actual ultrasound data.

This script:
  1. Auto-detects the US image ROI within the full frame (or uses --crop)
  2. Crops each frame to the ROI
  3. Applies pixel spacing (from CSV or YAML) to convert ROI pixels → mm
  4. Applies calibration: p_MUS = T_MUS_I @ p_I
  5. Applies per-frame tracking: p_C = T_C_MUS @ p_MUS
  6. Bin-averages into a voxel grid → NRRD

Deps: pip install numpy scipy Pillow pynrrd pyyaml tqdm
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import nrrd
except ImportError:
    sys.exit("pip install pynrrd")
try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")


# ── Math helpers ───────────────────────────────────────────────────────────

def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def pose_to_T(tx, ty, tz, qx, qy, qz, qw):
    return make_T(quat_to_R(qx, qy, qz, qw), np.array([tx, ty, tz]))


# ── Auto-detect US ROI within the full frame ───────────────────────────────

def detect_us_roi(frame_path, debug=False):
    """
    The 1920x1080 frame includes the Aixplorer UI chrome (dark borders,
    text overlays, depth scale, etc.). The actual US image is a bright
    rectangular region roughly in the center.

    Strategy:
      - Compute per-column and per-row mean intensity
      - Threshold to find the contiguous block with sustained brightness
      - Return (x0, y0, x1, y1) of the US image region

    Returns (x0, y0, x1, y1) in pixel coordinates.
    """
    img = np.array(Image.open(frame_path).convert('L'), dtype=np.float32)
    H, W = img.shape

    # Column profile: mean intensity per column
    col_mean = img.mean(axis=0)  # shape (W,)
    # Row profile: mean intensity per row
    row_mean = img.mean(axis=1)  # shape (H,)

    # Threshold: columns/rows with mean > some fraction of max
    # Use a low threshold since US images have lots of dark regions
    col_thresh = max(col_mean.max() * 0.08, 3.0)
    row_thresh = max(row_mean.max() * 0.08, 3.0)

    col_mask = col_mean > col_thresh
    row_mask = row_mean > row_thresh

    # Find contiguous runs — pick the longest one
    def longest_run(mask):
        starts, ends = [], []
        in_run = False
        for i, v in enumerate(mask):
            if v and not in_run:
                starts.append(i)
                in_run = True
            elif not v and in_run:
                ends.append(i)
                in_run = False
        if in_run:
            ends.append(len(mask))
        if not starts:
            return 0, len(mask)
        lengths = [e - s for s, e in zip(starts, ends)]
        best = np.argmax(lengths)
        return starts[best], ends[best]

    x0, x1 = longest_run(col_mask)
    y0, y1 = longest_run(row_mask)

    # Sanity: ROI should be at least 100px in each direction
    if (x1 - x0) < 100 or (y1 - y0) < 100:
        print(f"[roi]   Auto-detection found small ROI ({x1-x0}x{y1-y0}), using full frame")
        return 0, 0, W, H

    print(f"[roi]   Auto-detected US region: ({x0},{y0}) → ({x1},{y1})  "
          f"= {x1-x0} x {y1-y0} px")

    if debug:
        print(f"        Col threshold: {col_thresh:.1f}, Row threshold: {row_thresh:.1f}")
        print(f"        Col mean range: {col_mean.min():.1f}–{col_mean.max():.1f}")
        print(f"        Row mean range: {row_mean.min():.1f}–{row_mean.max():.1f}")

    return x0, y0, x1, y1


# ── Load calibration ───────────────────────────────────────────────────────

def load_calibration(yaml_path):
    """Load S_x, S_y, T_MUS_I from US_calibration.yaml."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    hus = raw['handheld']
    S_x = float(hus['S_x'])
    S_y = float(hus['S_y'])

    t_raw = hus['T_SI']
    if isinstance(t_raw, list):
        flat = np.array(t_raw, dtype=float).flatten()
        T = flat.reshape(4, 4)
    elif isinstance(t_raw, str):
        T = np.array(eval(t_raw), dtype=float).reshape(4, 4)
    else:
        raise ValueError(f"Unexpected T_SI type: {type(t_raw)}")

    print(f"[calib] YAML: S_x={S_x:.6f}  S_y={S_y:.6f} mm/px")
    print(f"[calib] T_MUS_I:\n{T}")
    return S_x, S_y, T


def load_csv_spacing(csv_path, volunteer="1", scan_group="H1"):
    """
    Parse US_scantype_availability.csv for pixel spacing.
    CSV row example:
      1;H1;URS01_H1;N;[1920, 1080];[0.0949559,0.09310205];...
    Returns (S_x, S_y) or None if not found.
    """
    if not csv_path or not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 6:
                continue
            urs = parts[0].strip()
            group = parts[1].strip()

            if urs == volunteer and group == scan_group:
                # Parse pixel spacing: "[0.0949559,0.09310205]"
                spacing_str = parts[5].strip()
                nums = re.findall(r'[\d.]+', spacing_str)
                if len(nums) >= 2:
                    sx, sy = float(nums[0]), float(nums[1])
                    print(f"[csv]   Pixel spacing for URS{urs}_{group}: "
                          f"S_x={sx:.6f}  S_y={sy:.6f} mm/px")
                    return sx, sy
    return None


# ── Load poses ─────────────────────────────────────────────────────────────

def load_poses(pose_path):
    """
    HUS_pose.txt: 15 values per row
      tx ty tz qx qy qz qw  tx_ref ... qw_ref  timestamp
    Returns (list[4x4], np.array timestamps).
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
                T = pose_to_T(vals[0], vals[1], vals[2],
                              vals[3], vals[4], vals[5], vals[6])
                transforms.append(T)
                timestamps.append(vals[14])
            elif len(vals) >= 8:
                T = pose_to_T(vals[0], vals[1], vals[2],
                              vals[3], vals[4], vals[5], vals[6])
                transforms.append(T)
                timestamps.append(vals[7])

    timestamps = np.array(timestamps)
    print(f"[poses] {len(transforms)} poses, "
          f"dt={timestamps[-1]-timestamps[0]:.2f}s")
    return transforms, timestamps


# ── Find and sort frames ──────────────────────────────────────────────────

def find_frames(data_dir):
    """Find PNGs, sort numerically (handles 0.png, 1.png, ... or 000000.png, ...)."""
    hus_dir = os.path.join(data_dir, 'HUS')
    if not os.path.isdir(hus_dir):
        hus_dir = data_dir

    files = glob.glob(os.path.join(hus_dir, '*.png'))
    if not files:
        files = glob.glob(os.path.join(hus_dir, '*.PNG'))
    if not files:
        return []

    # Sort numerically by the integer in the filename
    def sort_key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        nums = re.findall(r'\d+', stem)
        return int(nums[0]) if nums else 0

    files.sort(key=sort_key)
    print(f"[frames] {len(files)} PNGs in {hus_dir}")
    print(f"         first: {os.path.basename(files[0])}, "
          f"last: {os.path.basename(files[-1])}")
    return files


# ── Sync frames to poses ─────────────────────────────────────────────────

def sync(frame_paths, pose_ts, fps=15.0):
    """Match frames to poses by timestamp. Returns [(frame_idx, pose_idx)]."""
    n = len(frame_paths)

    # Extract frame number from filename
    indices = []
    for p in frame_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        nums = re.findall(r'\d+', stem)
        indices.append(int(nums[0]) if nums else len(indices))
    indices = np.array(indices, dtype=float)

    t0 = pose_ts[0]
    frame_ts = t0 + indices / fps

    matched = []
    for fi in range(n):
        diffs = np.abs(pose_ts - frame_ts[fi])
        pi = int(np.argmin(diffs))
        if diffs[pi] < 0.2:
            matched.append((fi, pi))

    print(f"[sync]  {len(matched)} / {n} frames matched")
    return matched


# ── Reconstruct ──────────────────────────────────────────────────────────

def reconstruct(frame_paths, matched, transforms, S_x, S_y, T_MUS_I,
                roi, voxel_mm=0.5, subsample=2):
    """
    Bin-average compounding.

    roi = (x0, y0, x1, y1) — the US image region within the full frame.
    Pixels are indexed relative to the ROI origin:
      p_I = [u_roi * S_x,  v_roi * S_y,  0,  1]
    """
    x0, y0, x1, y1 = roi
    roi_w = x1 - x0
    roi_h = y1 - y0
    print(f"[recon] ROI: ({x0},{y0})→({x1},{y1}) = {roi_w}×{roi_h} px")
    print(f"[recon] Physical: {roi_w*S_x:.1f} × {roi_h*S_y:.1f} mm")

    # Subsampled pixel grid within ROI
    us = np.arange(0, roi_w, subsample, dtype=np.float64)
    vs = np.arange(0, roi_h, subsample, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    npx = uu.size

    # Pixel → image coords (mm)
    pix_img = np.ones((npx, 4), dtype=np.float64)
    pix_img[:, 0] = uu.ravel() * S_x
    pix_img[:, 1] = vv.ravel() * S_y
    pix_img[:, 2] = 0.0

    # Pre-multiply calibration: p_MUS = T_MUS_I @ p_I
    pix_mk = (T_MUS_I @ pix_img.T).T  # (npx, 4)

    # Integer pixel indices for sampling the ROI
    uu_i = uu.ravel().astype(int)
    vv_i = vv.ravel().astype(int)

    # Image corners in marker frame (for bbox)
    corners = np.array([
        [0,               0,               0, 1],
        [(roi_w-1)*S_x,   0,               0, 1],
        [0,               (roi_h-1)*S_y,   0, 1],
        [(roi_w-1)*S_x,   (roi_h-1)*S_y,   0, 1],
    ])
    corners_mk = (T_MUS_I @ corners.T).T

    # ── Pass 1: bounding box ──
    print("[recon] pass 1 — bounding box")
    pts = []
    for fi, pi in tqdm(matched, desc="bbox", leave=False):
        c = (transforms[pi] @ corners_mk.T).T[:, :3]
        pts.append(c)
    pts = np.vstack(pts)

    pad = 5.0
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    dims = np.ceil((hi - lo) / voxel_mm).astype(int)
    nvox = int(np.prod(dims))
    print(f"[recon] bbox: {lo} → {hi}")
    print(f"[recon] grid: {dims}  "
          f"({dims[0]*voxel_mm:.1f}×{dims[1]*voxel_mm:.1f}×{dims[2]*voxel_mm:.1f} mm, "
          f"{nvox/1e6:.1f}M voxels)")

    if nvox > 2_000_000_000:
        sys.exit("[ERROR] Volume too large — increase --voxel_size")

    # ── Pass 2: scatter ──
    print("[recon] pass 2 — voxelizing")
    vol_sum = np.zeros(dims, dtype=np.float64)
    vol_count = np.zeros(dims, dtype=np.uint16)
    inv_v = 1.0 / voxel_mm
    d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])

    for fi, pi in tqdm(matched, desc="voxelize"):
        # Load full frame, crop to ROI
        full = np.array(Image.open(frame_paths[fi]).convert('L'))
        roi_img = full[y0:y1, x0:x1]

        # Sample intensities at subsampled positions
        intensities = roi_img[
            np.clip(vv_i, 0, roi_img.shape[0]-1),
            np.clip(uu_i, 0, roi_img.shape[1]-1)
        ].astype(np.float64)

        if intensities.mean() < 5.0:
            continue

        # Transform to world (camera frame)
        world = (transforms[pi] @ pix_mk.T).T[:, :3]

        # Voxel indices
        idx = ((world - lo) * inv_v).astype(np.int32)
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < d0) & \
             (idx[:, 1] >= 0) & (idx[:, 1] < d1) & \
             (idx[:, 2] >= 0) & (idx[:, 2] < d2)

        vi = idx[ok]
        iv = intensities[ok]

        flat = vi[:, 0] * (d1 * d2) + vi[:, 1] * d2 + vi[:, 2]
        np.add.at(vol_sum.ravel(), flat, iv)
        np.add.at(vol_count.ravel(), flat, 1)

    # Average + smooth
    mask = vol_count > 0
    vol = np.zeros(dims, dtype=np.float32)
    vol[mask] = (vol_sum[mask] / vol_count[mask]).astype(np.float32)
    vol = gaussian_filter(vol, sigma=0.5).astype(np.float32)

    pct = mask.sum() / max(mask.size, 1) * 100
    print(f"[recon] filled {mask.sum():,} / {mask.size:,} voxels ({pct:.1f}%)")

    return vol, lo, np.array([voxel_mm]*3)


# ── Save ──────────────────────────────────────────────────────────────────

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
    print(f"[save]  {path} ({mb:.1f} MB)")
    print(f"        shape={vol.shape}  spacing={spacing}  origin={origin}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HUS → 3D NRRD (CAVA)")
    ap.add_argument('--data_dir',   required=True)
    ap.add_argument('--calib_yaml', required=True)
    ap.add_argument('--output',     default=None)
    ap.add_argument('--csv_path',   default=None,
                    help='US_scantype_availability.csv (overrides YAML pixel spacing)')
    ap.add_argument('--volunteer',  default='1', help='URS number (default: 1)')
    ap.add_argument('--scan_group', default='H1', help='Scan group (default: H1)')
    ap.add_argument('--voxel_size', type=float, default=0.5)
    ap.add_argument('--subsample',  type=int, default=2)
    ap.add_argument('--crop', type=str, default=None,
                    help='Manual US ROI as x0,y0,x1,y1 (skip auto-detect)')
    args = ap.parse_args()

    # 1. calibration
    S_x, S_y, T_MUS_I = load_calibration(args.calib_yaml)

    # 2. override pixel spacing from CSV if available
    if args.csv_path:
        csv_sp = load_csv_spacing(args.csv_path, args.volunteer, args.scan_group)
        if csv_sp:
            S_x, S_y = csv_sp
            print(f"[info]  Using CSV pixel spacing: S_x={S_x:.6f} S_y={S_y:.6f}")

    # 3. frames
    frames = find_frames(args.data_dir)
    if not frames:
        sys.exit(f"[ERROR] No PNGs in {args.data_dir}")

    # 4. detect or set ROI
    if args.crop:
        roi = tuple(int(v) for v in args.crop.split(','))
        print(f"[roi]   Manual crop: {roi}")
    else:
        roi = detect_us_roi(frames[0])

    # 5. poses
    pose_path = os.path.join(args.data_dir, 'HUS_pose.txt')
    if not os.path.exists(pose_path):
        sys.exit(f"[ERROR] Not found: {pose_path}")
    transforms, timestamps = load_poses(pose_path)

    # 6. sync
    matched = sync(frames, timestamps, fps=15.0)
    if not matched:
        sys.exit("[ERROR] No frames matched")

    # 7. reconstruct
    print(f"\n{'='*60}")
    print(f"  URS{args.volunteer}_{args.scan_group}")
    print(f"  S_x={S_x:.6f}  S_y={S_y:.6f} mm/px")
    print(f"  voxel={args.voxel_size}mm  subsample={args.subsample}x")
    print(f"{'='*60}\n")

    vol, origin, spacing = reconstruct(
        frames, matched, transforms,
        S_x, S_y, T_MUS_I, roi,
        voxel_mm=args.voxel_size,
        subsample=args.subsample,
    )

    # 8. save
    out = args.output or os.path.join(args.data_dir, 'recon.nrrd')
    save_nrrd(vol, origin, spacing, out)
    print("[done]")


if __name__ == '__main__':
    main()