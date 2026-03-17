"""
plot_shadow_elongation_check.py

Diagnostic: for each bone candidate blob apply two filters:
  1. Elongation    — blob must be sufficiently line-like (PCA axis ratio)
  2. Shadow check  — only run on elongated blobs; pixels in a fixed window
                     immediately below the blob must be near-zero

Both must pass for a blob to be kept as a transverse process candidate.

Usage
-----
  python plot_shadow_elongation_check.py
  python plot_shadow_elongation_check.py --z 42
  python plot_shadow_elongation_check.py --z 42 --out debug.png
"""

import argparse
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
    """
    PCA on blob pixel coordinates.
    Returns (elongation_ratio, angle_deg, centre, half_axes)
      elongation_ratio : eigenvalue_major / eigenvalue_minor
      angle_deg        : orientation of major axis in degrees from horizontal
      centre           : (col, row) centroid
      half_axes        : (major, minor) half-lengths = 2*std for plotting
    """
    coords = np.stack([lat_idx, dep_idx], axis=1).astype(float)  # (N,2) col,row
    centre = coords.mean(axis=0)
    coords -= centre
    cov             = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    eigvals          = np.abs(eigvals)
    if eigvals[0] < 1e-6:
        return 1.0, 0.0, centre, (1.0, 1.0)
    ratio      = eigvals[1] / eigvals[0]
    major_vec  = eigvecs[:, 1]
    angle_deg  = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    half_axes  = (2 * np.sqrt(eigvals[1]), 2 * np.sqrt(eigvals[0]))
    return ratio, angle_deg, centre, half_axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume",           default=DEFAULT_VOLUME)
    parser.add_argument("--candidates",       default=DEFAULT_CANDIDATES)
    parser.add_argument("--z",                type=int,   default=None)
    parser.add_argument("--min-elongation",   type=float, default=MIN_ELONGATION)
    parser.add_argument("--near-zero-thresh", type=float, default=NEAR_ZERO_THRESH)
    parser.add_argument("--shadow-frac",      type=float, default=SHADOW_FRAC_NEEDED)
    parser.add_argument("--shadow-rows",      type=int,   default=SHADOW_ROWS)
    parser.add_argument("--out",              default=None)
    args = parser.parse_args()

    near_zero_thresh   = args.near_zero_thresh
    shadow_frac_needed = args.shadow_frac
    shadow_rows        = args.shadow_rows
    min_elongation     = args.min_elongation

    vol_full, _ = nrrd.read(args.volume)
    n_z  = vol_full.shape[2]
    z    = args.z if args.z is not None else n_z // 2

    img_sl  = load_slice(args.volume,     z)
    cand_sl = load_slice(args.candidates, z)

    slice_max = img_sl.max()
    thresh    = near_zero_thresh * slice_max

    vmax     = img_sl.max()
    img_norm = img_sl / vmax if vmax > 0 else img_sl

    n_rows, n_cols = img_sl.shape

    struct8    = generate_binary_structure(2, 2)
    labeled, n = label(cand_sl, structure=struct8)
    print(f"Z={z}  blobs={n}  shape={img_sl.shape}")
    print(f"  slice max={slice_max:.1f}  shadow thresh={thresh:.1f}")

    # ── per-blob decisions ────────────────────────────────────────────────
    blobs = {}
    for blob_id in range(1, n + 1):
        dep_idx, lat_idx = np.where(labeled == blob_id)

        if len(dep_idx) < MIN_BLOB_PIXELS:
            blobs[blob_id] = dict(
                elongated=False, shadow=False,
                ratio=1.0, frac_dark=0.0,
                deepest_row=int(dep_idx.max()),
                cols_at_deepest=lat_idx,
                centre=(float(lat_idx.mean()), float(dep_idx.mean())),
                half_axes=(1.0, 1.0), angle_deg=0.0,
            )
            continue

        # ── elongation (PCA) ──────────────────────────────────────────────
        ratio, angle_deg, centre, half_axes = blob_pca(dep_idx, lat_idx)
        is_elongated = ratio >= min_elongation

        # ── shadow check (only meaningful if elongated) ───────────────────
        deepest_row     = int(dep_idx.max())
        cols_at_deepest = lat_idx[dep_idx == deepest_row]
        if len(cols_at_deepest) == 0:
            cols_at_deepest = lat_idx

        row_start   = deepest_row + 1
        row_end     = min(row_start + shadow_rows, n_rows)
        sample_rows = slice(row_start, row_end)

        region = img_sl[sample_rows, :][:, cols_at_deepest]

        if region.size == 0:
            frac_dark, has_shadow = 0.0, False
        else:
            # near-zero: strictly above 0 (not out-of-frame) but below thresh
            is_shadow  = (region > 0) & (region < thresh)
            frac_dark  = is_shadow.sum() / region.size
            has_shadow = frac_dark >= shadow_frac_needed

        blobs[blob_id] = dict(
            elongated       = is_elongated,
            shadow          = has_shadow,
            ratio           = ratio,
            frac_dark       = frac_dark,
            deepest_row     = deepest_row,
            cols_at_deepest = cols_at_deepest,
            centre          = centre,
            half_axes       = half_axes,
            angle_deg       = angle_deg,
        )

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n{'blob':>5} {'deepest':>8} {'n_px':>6} "
          f"{'ratio':>7} {'elongated':>10} "
          f"{'frac_dark':>10} {'shadow':>8} {'result':>8}")
    for blob_id, b in blobs.items():
        dep_idx, _ = np.where(labeled == blob_id)
        kept = b['elongated'] and b['shadow']
        print(f"{blob_id:>5} {b['deepest_row']:>8} {len(dep_idx):>6} "
              f"{b['ratio']:>7.1f} {str(b['elongated']):>10} "
              f"{b['frac_dark']:>10.0%} {str(b['shadow']):>8} "
              f"{'KEEP' if kept else 'REJECT':>8}")

    n_kept = sum(b['elongated'] and b['shadow'] for b in blobs.values())
    print(f"\n  kept={n_kept}  rejected={n - n_kept}")

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
            col_c  = b['centre'][0]
            row_c  = b['centre'][1]
            col_min = int(b['cols_at_deepest'].min())
            col_max = int(b['cols_at_deepest'].max())
            a, bm  = b['half_axes']   # major, minor
            angle  = b['angle_deg']
            cos_a  = np.cos(np.radians(angle))
            sin_a  = np.sin(np.radians(angle))

            if panel == 0:
                # PCA ellipse
                color = "cyan" if b['elongated'] else "red"
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
                # shadow sampling window — only draw for elongated blobs
                if not b['elongated']:
                    continue
                color      = "cyan" if b['shadow'] else "red"
                col_center = int(b['cols_at_deepest'].mean())
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
                # final result
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
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()