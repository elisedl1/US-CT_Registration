"""
  python plot_shadow_elongation_check.py
  python plot_shadow_elongation_check.py --z 42
  python plot_shadow_elongation_check.py --z 42 --out debug.png
  python plot_shadow_elongation_check.py --z 42 --save-volume filtered.nrrd
  python plot_shadow_elongation_check.py --z 42 --save-volume ""   # skip volume
"""

import argparse
import os
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label, generate_binary_structure

_BASE = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/"
    "Registration/US_Vertebra_axial_cal/"
)
DEFAULT_VOLUME     = _BASE + "US_complete_cal_preprocessed.nrrd"
DEFAULT_CANDIDATES = _BASE + "US_complete_cal_bone_candidates.nrrd"

NEAR_ZERO_THRESH    = 0.10   # pixels below this fraction of slice max = shadow
SHADOW_FRAC_NEEDED  = 0.60   # fraction of sampled pixels that must be near-zero
SHADOW_ROWS         = 15     # how many rows immediately below blob to sample
MIN_ELONGATION      = 3.0    # min PCA eigenvalue ratio to be considered elongated
MIN_BLOB_PIXELS     = 10     # ignore tiny blobs


def load_slice(path, z, flip=True):
    vol, _ = nrrd.read(path)
    sl = vol[:, :, z].astype(np.float32).T
    if flip:
        sl = sl[::-1]
    return sl


def blob_pca(dep_idx, lat_idx):

    coords = np.stack([lat_idx, dep_idx], axis=1).astype(float)  # (N,2) col,row
    centre = coords.mean(axis=0)
    coords -= centre
    cov              = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    eigvals          = np.abs(eigvals)
    if eigvals[0] < 1e-6:
        return 1.0, 0.0, centre, (1.0, 1.0)
    ratio      = eigvals[1] / eigvals[0]
    major_vec  = eigvecs[:, 1]
    angle_deg  = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    half_axes  = (2 * np.sqrt(eigvals[1]), 2 * np.sqrt(eigvals[0]))
    return ratio, angle_deg, centre, half_axes


def filter_slice(img_sl, cand_sl, near_zero_thresh, shadow_frac_needed,
                 shadow_rows, min_elongation):
    """
    Apply elongation + shadow filter to one slice

    """
    n_rows    = img_sl.shape[0]
    slice_max = img_sl.max()
    thresh    = near_zero_thresh * slice_max

    struct8    = generate_binary_structure(2, 2)
    labeled, n = label(cand_sl, structure=struct8)

    blobs    = {}
    filtered = np.zeros_like(cand_sl, dtype=np.uint8)

    for blob_id in range(1, n + 1):
        dep_idx, lat_idx = np.where(labeled == blob_id)

        if len(dep_idx) < MIN_BLOB_PIXELS:
            blobs[blob_id] = dict(
                elongated=False, shadow=False,
                ratio=1.0, frac_dark=0.0,
                deepest_row=int(dep_idx.max()),
                col_min=int(lat_idx.min()),
                col_max=int(lat_idx.max()),
                centre=(float(lat_idx.mean()), float(dep_idx.mean())),
                half_axes=(1.0, 1.0), angle_deg=0.0,
            )
            continue

        # ── elongation (PCA) ──────────────────────────────────────────────
        ratio, angle_deg, centre, half_axes = blob_pca(dep_idx, lat_idx)
        is_elongated = ratio >= min_elongation

        # ── shadow check ──────────────────────────────────────────────────
        deepest_row = int(dep_idx.max())
        col_min     = int(lat_idx.min())
        col_max     = int(lat_idx.max()) + 1   

        row_start = deepest_row + 1
        row_end   = min(row_start + shadow_rows, n_rows)

        region = img_sl[row_start:row_end, col_min:col_max]

        if region.size == 0:
            frac_dark, has_shadow = 0.0, False
        else:
            # near-zero: strictly above 0 (not out-of-frame) but below thresh
            is_shadow  = (region > 0) & (region < thresh)
            frac_dark  = is_shadow.sum() / region.size
            has_shadow = frac_dark >= shadow_frac_needed

        if is_elongated and has_shadow:
            filtered[labeled == blob_id] = 1

        blobs[blob_id] = dict(
            elongated  = is_elongated,
            shadow     = has_shadow,
            ratio      = ratio,
            frac_dark  = frac_dark,
            deepest_row= deepest_row,
            col_min    = col_min,
            col_max    = col_max - 1,   # store inclusive for display
            centre     = centre,
            half_axes  = half_axes,
            angle_deg  = angle_deg,
        )

    return filtered, blobs


def save_filtered_volume(vol_path, cand_path, out_path,
                         near_zero_thresh, shadow_frac_needed,
                         shadow_rows, min_elongation):
    """
    Run the elongation + shadow filter across every Z slice of the candidate
    """
    print(f"\nBuilding filtered volume over all Z slices …")
    print(f"  volume     : {vol_path}")
    print(f"  candidates : {cand_path}")

    vol_raw,  _        = nrrd.read(vol_path)
    cand_raw, cand_hdr = nrrd.read(cand_path)

    # Normalise intensity to [0, 1] — same as the plotting path
    vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
    vol_f = (vol_raw.astype(np.float32) - vmin) / (vmax - vmin + 1e-9)

    n_z     = vol_f.shape[2]
    out_vol = np.zeros(vol_f.shape, dtype=np.uint8)

    for z in range(n_z):
        if z % 20 == 0:
            print(f"  {z}/{n_z}", end="\r", flush=True)

        # Apply the same flip that load_slice uses so geometry is consistent
        img_sl  = vol_f[:, :, z].astype(np.float32).T[::-1]
        cand_sl = cand_raw[:, :, z].astype(np.float32).T[::-1]

        filtered_sl, _ = filter_slice(
            img_sl, cand_sl,
            near_zero_thresh, shadow_frac_needed,
            shadow_rows, min_elongation,
        )

        # Undo the flip before storing back into volume axes
        out_vol[:, :, z] = filtered_sl[::-1].T

    n_in  = int(cand_raw.astype(bool).sum())
    n_out = int(out_vol.sum())
    print(f"\n  kept voxels : {n_out} / {n_in}  ({100*n_out/max(n_in,1):.1f}%)")

    out_hdr = cand_hdr.copy()
    out_hdr["type"] = "unsigned char"
    out_hdr.setdefault("encoding", "gzip")

    print(f"  saving     : {out_path}")
    nrrd.write(out_path, out_vol, out_hdr)
    print("  done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume",           default=DEFAULT_VOLUME)
    parser.add_argument("--candidates",       default=DEFAULT_CANDIDATES)
    parser.add_argument("--z",                type=int,   default=None)
    parser.add_argument("--min-elongation",   type=float, default=MIN_ELONGATION)
    parser.add_argument("--near-zero-thresh", type=float, default=NEAR_ZERO_THRESH)
    parser.add_argument("--shadow-frac",      type=float, default=SHADOW_FRAC_NEEDED)
    parser.add_argument("--shadow-rows",      type=int,   default=SHADOW_ROWS)
    parser.add_argument("--out",              default=None,
                        help="Save plot to this path instead of showing it")
    parser.add_argument("--save-volume",      default=None,
                        metavar="PATH",
                        help="Save full filtered 3-D NRRD to this path. "
                             "Defaults to <candidates_stem>_filtered.nrrd "
                             "next to the candidates file. "
                             "Pass empty string to skip.")
    args = parser.parse_args()

    near_zero_thresh   = args.near_zero_thresh
    shadow_frac_needed = args.shadow_frac
    shadow_rows        = args.shadow_rows
    min_elongation     = args.min_elongation

    # ── resolve save-volume path ──────────────────────────────────────────
    # Default: <dir of candidates>/<stem>_filtered.nrrd
    save_volume = args.save_volume
    if save_volume is None:
        cand_stem   = os.path.splitext(args.candidates)[0]
        save_volume = cand_stem + "_filtered.nrrd"

    # ── optional: build + save filtered volume across all Z ──────────────
    if save_volume:
        save_filtered_volume(
            args.volume, args.candidates, save_volume,
            near_zero_thresh, shadow_frac_needed,
            shadow_rows, min_elongation,
        )

    # ── load the single diagnostic slice ─────────────────────────────────
    vol_full, _ = nrrd.read(args.volume)
    n_z  = vol_full.shape[2]
    z    = args.z if args.z is not None else n_z // 2

    img_sl  = load_slice(args.volume,     z)
    cand_sl = load_slice(args.candidates, z)

    slice_max = img_sl.max()
    thresh    = near_zero_thresh * slice_max
    vmax      = img_sl.max()
    img_norm  = img_sl / vmax if vmax > 0 else img_sl
    n_rows, _ = img_sl.shape

    _, blobs = filter_slice(
        img_sl, cand_sl,
        near_zero_thresh, shadow_frac_needed,
        shadow_rows, min_elongation,
    )
    n = len(blobs)

    # ── print table ───────────────────────────────────────────────────────
    print(f"\nZ={z}  blobs={n}  shape={img_sl.shape}")
    print(f"  slice max={slice_max:.1f}  shadow thresh={thresh:.1f}")
    print(f"\n{'blob':>5} {'deepest':>8} {'ratio':>7} {'elongated':>10} "
          f"{'frac_dark':>10} {'shadow':>8} {'result':>8}")
    for blob_id, b in blobs.items():
        kept = b['elongated'] and b['shadow']
        print(f"{blob_id:>5} {b['deepest_row']:>8} "
              f"{b['ratio']:>7.1f} {str(b['elongated']):>10} "
              f"{b['frac_dark']:>10.0%} {str(b['shadow']):>8} "
              f"{'KEEP' if kept else 'REJECT':>8}")

    n_kept = sum(b['elongated'] and b['shadow'] for b in blobs.values())
    print(f"\n  kept={n_kept}  rejected={n - n_kept}")

    # ── labeled mask needed for the display slice blob overlay ───────────
    struct8    = generate_binary_structure(2, 2)
    labeled, _ = label(cand_sl, structure=struct8)

    # ── plot: 3 panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"Z={z}  |  elongation ratio≥{min_elongation:.1f}  |  "
        f"shadow: >{near_zero_thresh:.0%} of max={slice_max:.1f} "
        f"thresh={thresh:.1f}, need {shadow_frac_needed:.0%} dark "
        f"in {shadow_rows} rows below blob",
        fontsize=10
    )

    titles = [
        "Elongation check\ncyan=pass  red=fail  (PCA ellipse)",
        "Shadow check (elongated blobs only)\ncyan=pass  red=fail",
        "Final result\ncyan=KEEP  red=REJECT",
    ]

    for panel, ax in enumerate(axes):
        ax.imshow(img_norm, cmap="gray", origin="upper", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(cand_sl == 0, cand_sl),
                  cmap="autumn", alpha=0.5, origin="upper")

        for blob_id, b in blobs.items():
            col_c   = b['centre'][0]
            row_c   = b['centre'][1]
            col_min = b['col_min']
            col_max = b['col_max']
            a, bm   = b['half_axes']
            angle   = b['angle_deg']
            cos_a   = np.cos(np.radians(angle))
            sin_a   = np.sin(np.radians(angle))

            if panel == 0:
                color   = "cyan" if b['elongated'] else "red"
                t       = np.linspace(0, 2 * np.pi, 60)
                ell_col = col_c + a * np.cos(t) * cos_a - bm * np.sin(t) * sin_a
                ell_row = row_c + a * np.cos(t) * sin_a + bm * np.sin(t) * cos_a
                ax.plot(ell_col, ell_row, color=color, lw=1.5, alpha=0.8)
                ax.plot([col_c - a * cos_a, col_c + a * cos_a],
                        [row_c - a * sin_a, row_c + a * sin_a],
                        color=color, lw=1, alpha=0.6, linestyle="--")
                ax.text(col_c, row_c - bm - 3, f"{b['ratio']:.1f}",
                        color=color, fontsize=6, ha="center", va="bottom")

            elif panel == 1:
                if not b['elongated']:
                    continue
                color      = "cyan" if b['shadow'] else "red"
                col_center = (col_min + col_max) // 2
                row_start  = b['deepest_row'] + 1
                row_end    = min(row_start + shadow_rows, n_rows) - 1
                ax.plot([col_min, col_max], [b['deepest_row'], b['deepest_row']],
                        color=color, lw=1, alpha=0.5, linestyle=":")
                ax.plot([col_min, col_max], [row_end, row_end],
                        color=color, lw=2, alpha=0.8)
                ax.plot([col_center, col_center],
                        [b['deepest_row'], row_end],
                        color=color, lw=1.2, alpha=0.7, linestyle="--")
                ax.text(col_center, row_end + 2, f"{b['frac_dark']:.0%}",
                        color=color, fontsize=6, ha="center", va="top")

            else:
                kept  = b['elongated'] and b['shadow']
                color = "cyan" if kept else "red"
                ax.plot(col_c, row_c, marker="+", color=color,
                        markersize=10, markeredgewidth=1.5)
                if not kept:
                    reasons = []
                    if not b['elongated']:
                        reasons.append(f"ratio={b['ratio']:.1f}")
                    if not b['shadow']:
                        reasons.append(f"dark={b['frac_dark']:.0%}")
                    ax.text(col_c, row_c - 4, "\n".join(reasons),
                            color=color, fontsize=5, ha="center", va="bottom")

        ax.set_xlabel("lateral (cols)")
        ax.set_ylabel("depth (rows, 0 = transducer)")
        ax.set_title(titles[panel], fontsize=9)

    patches = [
        mpatches.Patch(color="orange", alpha=0.6, label="bone candidates"),
        mpatches.Patch(color="cyan",   alpha=0.8, label="pass / keep"),
        mpatches.Patch(color="red",    alpha=0.8, label="fail / reject"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()