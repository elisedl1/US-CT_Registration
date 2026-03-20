"""
Acoustic Shadow Enhancement (ASE) filter

Usage:
  python ase_filter.py
  python ase_filter.py --save-ase
  python ase_filter.py --z 42 --out preview.png
"""

import argparse
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

_BASE = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/"
    "Registration/US_Vertebra_axial_cal/"
)
DEFAULT_VOLUME  = _BASE + "US_complete_cal_preprocessed.nrrd"
DEFAULT_MASK    = _BASE + "US_complete_cal_mask.nrrd"
DEFAULT_ASE_OUT = _BASE + "US_complete_cal_ase_enhanced.nrrd"


def load_slice(path, z, flip=True):
    vol, _ = nrrd.read(path)
    sl = vol[:, :, z].astype(np.float32).T
    if flip:
        sl = sl[::-1]
    return sl


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10) if hi > lo else np.zeros_like(x, dtype=np.float32)


def speckle_reduce(img, window=3):
    img        = img.astype(np.float64)
    local_mean = uniform_filter(img, size=window)
    local_sq   = uniform_filter(img ** 2, size=window)
    local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)
    global_var = np.var(img)
    weight     = local_var / (local_var + global_var + 1e-10)
    return norm01(local_mean + weight * (img - local_mean)).astype(np.float32)


def acoustic_shadow_energy(img):
    rows     = img.shape[0]
    cumsum   = np.cumsum(img, axis=0)
    row_idx  = np.arange(1, rows + 1).reshape(-1, 1)
    cum_mean = norm01(cumsum / row_idx)
    shadow   = 1.0 - cum_mean
    # prod     = img * shadow
    prod = 0.6 * (img * shadow) + (1 - 0.6) * img # alpha - = 0.6
    enhanced = np.where(prod <= 0.5,
                        2 * prod**2,
                        1 - 2 * (1 - prod)**2)
    return enhanced.astype(np.float32), shadow.astype(np.float32)


# ── Save ASE-enhanced volume ──────────────────────────────────────────────────
def save_ase_volume(vol_path, mask_path, out_path):
    print(f"\nBuilding ASE-enhanced volume ...")
    print(f"  input  : {vol_path}")
    print(f"  mask   : {mask_path}")

    vol_raw,  header = nrrd.read(vol_path)
    mask_raw, _      = nrrd.read(mask_path)

    vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
    vol_f = (vol_raw.astype(np.float32) - vmin) / (vmax - vmin + 1e-9)

    n_z     = vol_f.shape[2]
    ase_vol = np.zeros_like(vol_f, dtype=np.float32)

    for z in range(n_z):
        if z % 20 == 0:
            print(f"  {z}/{n_z}", end="\r", flush=True)

        img_sl  = vol_f[:, :, z].astype(np.float32).T[::-1]
        mask_sl = mask_raw[:, :, z].astype(np.float32).T[::-1]

        img_sl   = norm01(img_sl) * (mask_sl > 0)
        filtered = speckle_reduce(img_sl)
        enhanced, _ = acoustic_shadow_energy(filtered)

        ase_vol[:, :, z] = enhanced[::-1].T

    print(f"\n  range  : [{ase_vol.min():.4f}, {ase_vol.max():.4f}]")

    out_hdr = header.copy()
    out_hdr["type"] = "float"
    out_hdr.setdefault("encoding", "gzip")

    print(f"  saving : {out_path}")
    nrrd.write(out_path, ase_vol, out_hdr)
    print("  done.")


# ── Single-slice preview plot ─────────────────────────────────────────────────
def plot_preview(z, img_sl, mask_sl, out_path=None):
    img_sl   = norm01(img_sl) * (mask_sl > 0)
    filtered = speckle_reduce(img_sl)
    enhanced, shadow = acoustic_shadow_energy(filtered)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"ASE filter preview  |  Z={z}", fontsize=10)

    for ax, img, cmap, title in zip(axes, 
        [img_sl,   shadow,              enhanced],
        ["gray",   "hot",               "gray"],
        ["Original (masked)", "Shadow weight\n(1 − cumul mean)", "ASE enhanced"]):
        ax.imshow(norm01(img), cmap=cmap, origin="upper", aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lateral (cols)")
    axes[0].set_ylabel("depth (rows, 0 = transducer)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--volume",   "-v", default=DEFAULT_VOLUME)
    parser.add_argument("--mask",     "-m", default=DEFAULT_MASK)
    parser.add_argument("--z",              type=int, default=None)
    parser.add_argument("--out",            default=None,
                        help="Save preview plot to this path")
    parser.add_argument("--save-ase",       action="store_true",
                        help="Save ASE-enhanced volume as NRRD")
    parser.add_argument("--ase-out",        default=None,
                        help="Override ASE output path (default: <input>_ase_enhanced.nrrd)")
    args = parser.parse_args()

    if args.save_ase:
        import os
        ase_out = args.ase_out or os.path.splitext(args.volume)[0] + "_ase_enhanced.nrrd"
        save_ase_volume(args.volume, args.mask, ase_out)

    vol_raw, _ = nrrd.read(args.volume)
    n_z = vol_raw.shape[2]
    z   = args.z if args.z is not None else n_z // 2
    print(f"Volume shape : {vol_raw.shape}   Z={z}")

    img_sl  = load_slice(args.volume, z)
    mask_sl = load_slice(args.mask,   z)

    plot_preview(z, img_sl, mask_sl, out_path=args.out)


if __name__ == "__main__":
    main()