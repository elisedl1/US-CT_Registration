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
DEFAULT_CANDIDATES = _BASE + "US_complete_cal_preprocessed_ase_enhanced_overlap_filtered_binary.nrrd"

NEAR_ZERO_THRESH    = 0.10   # pixels below this fraction of slice max = shadow
SHADOW_FRAC_NEEDED  = 0.60   # fraction of sampled pixels that must be near-zero
SHADOW_ROWS         = 15     # how many rows immediately below blob to sample
MIN_ELONGATION      = 8     # min PCA eigenvalue ratio to be considered elongated
MIN_BLOB_PIXELS     = 10     # ignore tiny blobs
MIN_BONE_WIDTH      = 0     # min bounding-box width (cols) to be kept as bone
MIN_MINOR_AXIS      = 3.0    # min PCA minor half-axis (px); rejects tiny slivers


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
                 shadow_rows, min_elongation, min_bone_width, min_minor_axis):
    """
    Apply elongation + minor-axis size + shadow + width filter to one slice.

    Paired-blob rule: if exactly 2 blobs pass the full elongation+thickness
    check (ratio >= min_elongation AND minor_axis >= min_minor_axis), both are
    kept regardless of shadow or width — the presence of a matching pair is
    strong evidence of a vertebra surface pair.
    """
    n_rows    = img_sl.shape[0]
    slice_max = img_sl.max()
    thresh    = near_zero_thresh * slice_max

    struct8    = generate_binary_structure(2, 2)
    labeled, n = label(cand_sl, structure=struct8)

    blobs    = {}
    filtered = np.zeros_like(cand_sl, dtype=np.uint8)

    # ── first pass: compute all blob properties ───────────────────────────
    for blob_id in range(1, n + 1):
        dep_idx, lat_idx = np.where(labeled == blob_id)

        if len(dep_idx) < MIN_BLOB_PIXELS:
            blobs[blob_id] = dict(
                elongated=False, shadow=False, wide_enough=False,
                thick_enough=False, paired=False,
                ratio=1.0, frac_dark=0.0, width=0, minor_axis=0.0,
                deepest_row=int(dep_idx.max()),
                col_min=int(lat_idx.min()),
                col_max=int(lat_idx.max()),
                centre=(float(lat_idx.mean()), float(dep_idx.mean())),
                half_axes=(1.0, 1.0), angle_deg=0.0,
            )
            continue

        # ── elongation + minor axis (PCA) ─────────────────────────────────
        ratio, angle_deg, centre, half_axes = blob_pca(dep_idx, lat_idx)
        minor_axis      = half_axes[1]
        is_elongated    = ratio >= min_elongation
        is_thick_enough = minor_axis >= min_minor_axis

        # ── shadow check ──────────────────────────────────────────────────
        deepest_row = int(dep_idx.max())
        col_min     = int(lat_idx.min())
        col_max     = int(lat_idx.max()) + 1

        row_start = deepest_row + 1
        row_end   = min(row_start + shadow_rows, n_rows)
        region    = img_sl[row_start:row_end, col_min:col_max]

        if region.size == 0:
            frac_dark, has_shadow = 0.0, False
        else:
            is_shadow  = (region > 0) & (region < thresh)
            frac_dark  = is_shadow.sum() / region.size
            has_shadow = frac_dark >= shadow_frac_needed

        # ── width check ───────────────────────────────────────────────────
        blob_width     = col_max - col_min
        is_wide_enough = blob_width >= min_bone_width

        blobs[blob_id] = dict(
            elongated    = is_elongated,
            thick_enough = is_thick_enough,
            shadow       = has_shadow,
            wide_enough  = is_wide_enough,
            paired       = False,             # filled in below
            ratio        = ratio,
            minor_axis   = minor_axis,
            frac_dark    = frac_dark,
            deepest_row  = deepest_row,
            col_min      = col_min,
            col_max      = col_max - 1,       # store inclusive for display
            width        = blob_width,
            centre       = centre,
            half_axes    = half_axes,
            angle_deg    = angle_deg,
        )

    # ── paired-blob rule ──────────────────────────────────────────────────
    # If exactly 2 blobs pass the full elongation+thickness check, mark them
    # both as paired — they will be kept regardless of shadow / width.
    fully_elongated = [
        bid for bid, b in blobs.items()
        if b['elongated'] and b['thick_enough']
    ]
    paired_rescue = (len(fully_elongated) == 2)
    if paired_rescue:
        for bid in fully_elongated:
            blobs[bid]['paired'] = True

    # ── second pass: build filtered mask ─────────────────────────────────
    for blob_id, b in blobs.items():
        keep_normal = (b['elongated'] and b['thick_enough']
                       and b['shadow'] and b['wide_enough'])
        keep_paired = b['paired']
        if keep_normal or keep_paired:
            filtered[labeled == blob_id] = 1

    return filtered, blobs, paired_rescue


def save_filtered_volume(vol_path, cand_path, out_path,
                         near_zero_thresh, shadow_frac_needed,
                         shadow_rows, min_elongation, min_bone_width,
                         min_minor_axis):
    """
    Run the filter across every Z slice of the candidate volume.
    """
    print(f"\nBuilding filtered volume over all Z slices …")
    print(f"  volume     : {vol_path}")
    print(f"  candidates : {cand_path}")

    vol_raw,  _        = nrrd.read(vol_path)
    cand_raw, cand_hdr = nrrd.read(cand_path)

    vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
    vol_f = (vol_raw.astype(np.float32) - vmin) / (vmax - vmin + 1e-9)

    n_z     = vol_f.shape[2]
    out_vol = np.zeros(vol_f.shape, dtype=np.uint8)

    for z in range(n_z):
        if z % 20 == 0:
            print(f"  {z}/{n_z}", end="\r", flush=True)

        img_sl  = vol_f[:, :, z].astype(np.float32).T[::-1]
        cand_sl = cand_raw[:, :, z].astype(np.float32).T[::-1]

        filtered_sl, _, _ = filter_slice(
            img_sl, cand_sl,
            near_zero_thresh, shadow_frac_needed,
            shadow_rows, min_elongation, min_bone_width, min_minor_axis,
        )

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
    parser.add_argument("--min-bone-width",   type=int,   default=MIN_BONE_WIDTH,
                        help="Min blob bounding-box width (px) to be kept as bone")
    parser.add_argument("--min-minor-axis",   type=float, default=MIN_MINOR_AXIS,
                        help="Min PCA minor half-axis (px); rejects thin slivers")
    parser.add_argument("--out",              default=None,
                        help="Save plot to this path instead of showing it")
    parser.add_argument("--save-volume",      default=None,
                        metavar="PATH",
                        help="Save full filtered 3-D NRRD to this path. "
                             "Defaults to <candidates_stem>_filtered.nrrd. "
                             "Pass empty string to skip.")
    args = parser.parse_args()

    near_zero_thresh   = args.near_zero_thresh
    shadow_frac_needed = args.shadow_frac
    shadow_rows        = args.shadow_rows
    min_elongation     = args.min_elongation
    min_bone_width     = args.min_bone_width
    min_minor_axis     = args.min_minor_axis

    # ── resolve save-volume path ──────────────────────────────────────────
    save_volume = args.save_volume
    if save_volume is None:
        cand_stem   = os.path.splitext(args.candidates)[0]
        save_volume = cand_stem + "_filtered.nrrd"

    if save_volume:
        save_filtered_volume(
            args.volume, args.candidates, save_volume,
            near_zero_thresh, shadow_frac_needed,
            shadow_rows, min_elongation, min_bone_width, min_minor_axis,
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

    _, blobs, paired_rescue = filter_slice(
        img_sl, cand_sl,
        near_zero_thresh, shadow_frac_needed,
        shadow_rows, min_elongation, min_bone_width, min_minor_axis,
    )
    n = len(blobs)

    # ── print table ───────────────────────────────────────────────────────
    print(f"\nZ={z}  blobs={n}  shape={img_sl.shape}"
          f"{'  [PAIRED-BLOB RESCUE ACTIVE]' if paired_rescue else ''}")
    print(f"  slice max={slice_max:.1f}  shadow thresh={thresh:.1f}")
    print(f"\n{'blob':>5} {'deepest':>8} {'ratio':>7} {'minor':>7} "
          f"{'elongated':>10} {'thick':>7} {'frac_dark':>10} "
          f"{'shadow':>8} {'paired':>7} {'result':>8}")
    for blob_id, b in blobs.items():
        kept = ((b['elongated'] and b['thick_enough'] and b['shadow'] and b['wide_enough'])
                or b['paired'])
        print(f"{blob_id:>5} {b['deepest_row']:>8} "
              f"{b['ratio']:>7.1f} {b['minor_axis']:>7.1f} "
              f"{str(b['elongated']):>10} {str(b['thick_enough']):>7} "
              f"{b['frac_dark']:>10.0%} {str(b['shadow']):>8} "
              f"{str(b['paired']):>7} "
              f"{'KEEP' if kept else 'REJECT':>8}")

    n_kept = sum(
        (b['elongated'] and b['thick_enough'] and b['shadow'] and b['wide_enough'])
        or b['paired']
        for b in blobs.values()
    )
    print(f"\n  kept={n_kept}  rejected={n - n_kept}"
          f"{'  (paired-blob rescue)' if paired_rescue else ''}")

    # ── labeled mask needed for the display slice blob overlay ───────────
    struct8    = generate_binary_structure(2, 2)
    labeled, _ = label(cand_sl, structure=struct8)

    # ── plot: 3 panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    rescue_note = "  ★ PAIRED-BLOB RESCUE" if paired_rescue else ""
    fig.suptitle(
        f"Z={z}  |  elongation ratio≥{min_elongation:.1f}  minor axis≥{min_minor_axis:.1f}px  |  "
        f"shadow: >{near_zero_thresh:.0%} of max={slice_max:.1f} "
        f"thresh={thresh:.1f}, need {shadow_frac_needed:.0%} dark "
        f"in {shadow_rows} rows below blob  |  min width={min_bone_width}px"
        f"{rescue_note}",
        fontsize=9
    )

    titles = [
        "Elongation + thickness check\ncyan=pass  red=fail  (PCA ellipse)",
        "Shadow check (elongated+thick blobs only)\ncyan=pass  red=fail",
        "Final result\ncyan=KEEP  red=REJECT  gold=paired-rescue",
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
                color = "cyan" if (b['elongated'] and b['thick_enough']) else "red"
                t       = np.linspace(0, 2 * np.pi, 60)
                ell_col = col_c + a * np.cos(t) * cos_a - bm * np.sin(t) * sin_a
                ell_row = row_c + a * np.cos(t) * sin_a + bm * np.sin(t) * cos_a
                ax.plot(ell_col, ell_row, color=color, lw=1.5, alpha=0.8)
                ax.plot([col_c - a * cos_a, col_c + a * cos_a],
                        [row_c - a * sin_a, row_c + a * sin_a],
                        color=color, lw=1, alpha=0.6, linestyle="--")
                ax.text(col_c, row_c - bm - 3,
                        f"{b['ratio']:.1f} / {b['minor_axis']:.1f}px",
                        color=color, fontsize=6, ha="center", va="bottom")

            elif panel == 1:
                if not (b['elongated'] and b['thick_enough']):
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
                keep_normal = (b['elongated'] and b['thick_enough']
                               and b['shadow'] and b['wide_enough'])
                keep_paired = b['paired']
                kept  = keep_normal or keep_paired
                # gold for paired-rescue, cyan for normal keep, red for reject
                color = "gold" if keep_paired else ("cyan" if kept else "red")
                ax.plot(col_c, row_c, marker="+", color=color,
                        markersize=10, markeredgewidth=1.5)
                if not kept:
                    reasons = []
                    if not b['elongated']:
                        reasons.append(f"ratio={b['ratio']:.1f}")
                    if not b['thick_enough']:
                        reasons.append(f"minor={b['minor_axis']:.1f}px")
                    if not b['shadow']:
                        reasons.append(f"dark={b['frac_dark']:.0%}")
                    if not b['wide_enough']:
                        reasons.append(f"width={b['width']}px")
                    ax.text(col_c, row_c - 4, "\n".join(reasons),
                            color=color, fontsize=5, ha="center", va="bottom")
                elif keep_paired and not keep_normal:
                    ax.text(col_c, row_c - 4, "paired",
                            color=color, fontsize=5, ha="center", va="bottom")

        ax.set_xlabel("lateral (cols)")
        ax.set_ylabel("depth (rows, 0 = transducer)")
        ax.set_title(titles[panel], fontsize=9)

    patches = [
        mpatches.Patch(color="orange", alpha=0.6, label="bone candidates"),
        mpatches.Patch(color="cyan",   alpha=0.8, label="pass / keep"),
        mpatches.Patch(color="gold",   alpha=0.8, label="paired-rescue"),
        mpatches.Patch(color="red",    alpha=0.8, label="fail / reject"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()