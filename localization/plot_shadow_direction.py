"""
plot_shadow_direction.py

Diagnostic: pick one Z-slice, show the preprocessed US image with the bone
candidate blobs overlaid, and draw an arrow from each blob's deepest point
downward (in the direction the shadow check will look).

This lets you confirm:
  - which axis is "depth" (the direction ultrasound travels)
  - that "below" (deeper tissue) is in the +row direction

Usage
-----
  python plot_shadow_direction.py              # uses middle Z slice
  python plot_shadow_direction.py --z 42       # specific slice index
  python plot_shadow_direction.py --z 42 --out debug_slice.png
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
DEFAULT_MASK       = _BASE + "US_complete_cal_mask.nrrd"

SHADOW_DEPTH_FRAC  = 0.25   # same default as detect_bone_shadows.py
SHADOW_MIN_DEPTH   = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume",     default=DEFAULT_VOLUME)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--mask",       default=DEFAULT_MASK)
    parser.add_argument("--z",          type=int, default=None,
                        help="Z slice index to plot (default: middle slice)")
    parser.add_argument("--out",        default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    vol,   _  = nrrd.read(args.volume)
    cands, _  = nrrd.read(args.candidates)
    mask,  _  = nrrd.read(args.mask)

    # Shape is (rows, cols, Z)
    n_z = vol.shape[2]
    z   = args.z if args.z is not None else n_z // 2
    print(f"Volume shape : {vol.shape}  — plotting Z slice {z}/{n_z-1}")
    print(f"  axis-0 = rows  ({vol.shape[0]} px)  ← treated as DEPTH")
    print(f"  axis-1 = cols  ({vol.shape[1]} px)  ← treated as LATERAL")
    print(f"  axis-2 = Z     ({vol.shape[2]} frames) ← axial sweep")

    # Transpose: (rows=depth, cols=lateral) → (lateral, depth) so imshow
    # shows lateral on the horizontal axis and depth on the vertical axis.
    img_sl   = vol[:, :, z].astype(np.float32).T[::-1]
    cand_sl  = cands[:, :, z].T[::-1]
    mask_sl  = mask[:, :, z].T[::-1]

    # Normalise for display
    vmax = img_sl.max()
    img_norm = img_sl / vmax if vmax > 0 else img_sl

    rows, cols = img_sl.shape

    # Label blobs
    struct8      = generate_binary_structure(2, 2)
    labeled, n   = label(cand_sl, structure=struct8)
    print(f"  Candidate blobs in this slice: {n}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Z slice {z}  —  shadow direction diagnostic", fontsize=13)

    for ax, show_arrows in zip(axes, [False, True]):
        ax.imshow(img_norm, cmap="gray", origin="upper",
                  vmin=0, vmax=1, aspect="equal")
        ax.imshow(np.ma.masked_where(cand_sl == 0, cand_sl),
                  cmap="autumn", alpha=0.6, origin="upper")
        ax.imshow(np.ma.masked_where(mask_sl == 0, mask_sl * 0.15),
                  cmap="cool", alpha=0.2, origin="upper")

        if show_arrows:
            n_rows, n_cols = cand_sl.shape  # (depth, lateral)
            for blob_id in range(1, n + 1):
                dep_idx, lat_idx = np.where(labeled == blob_id)  # row=depth, col=lateral

                for lat in np.unique(lat_idx):
                    deepest_dep = int(dep_idx[lat_idx == lat].max())
                    depth_left  = n_rows - deepest_dep - 1
                    look_depth  = max(SHADOW_MIN_DEPTH,
                                    int(depth_left * SHADOW_DEPTH_FRAC))
                    arrow_len   = min(look_depth, depth_left)

                    ax.annotate(
                        "", xy=(lat, deepest_dep + arrow_len),
                        xytext=(lat, deepest_dep),
                        arrowprops=dict(arrowstyle="->", color="cyan",
                                        lw=1.0, mutation_scale=8),
                    )



        title = ("Left: US + candidates (no arrows)\n"
                 "Row 0 = top = near transducer\n"
                 "Row N = bottom = deeper tissue"
                 if not show_arrows else
                 "Right: same + shadow-search arrows\n"
                 "Cyan arrows = direction & depth of shadow check\n"
                 "(should point AWAY from transducer)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lateral (cols)")
        ax.set_ylabel("depth (rows, 0 = transducer)")

    patches = [
        mpatches.Patch(color="orange",  alpha=0.7, label="bone candidates"),
        mpatches.Patch(color="cyan",    alpha=0.7, label="shadow search direction"),
        mpatches.Patch(color="skyblue", alpha=0.4, label="image mask"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved : {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()