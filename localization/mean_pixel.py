"""
Mean Pixel Intensity (MI) + SRRM feature extraction for PSRE blobs.
Implements Shajudeen & Righetti (Med. Phys. 2017), Section 2.C.1 & 2.C.2.

Pipeline:
  1. Compute ASE — Eq. (7)
  2. Compute f * ASE product image
  3. Nonlinear contrast stretch -> mu(x,y) — Eq. (8)
  4. MI per blob — Eq. (9)
  5. SRRM SSR per blob — Eq. (10)

Inputs:
  - preprocessed US volume  (float32, [0,1])
  - binarized PSRE volume   (uint8)

Outputs:
  - _ase.nrrd           : ASE image
  - _f_ase.nrrd         : raw f * ASE product (Fig 5b equivalent)
  - _enhanced.nrrd      : mu(x,y) contrast-stretched image
  - _mi_map.nrrd        : per-blob MI values mapped spatially
  - _srrm_map.nrrd      : per-blob SSR values mapped spatially
"""

import numpy as np
import nrrd
from scipy.ndimage import label
from scipy.optimize import curve_fit


# ─────────────────────────── ASE — Eq. (7) ───────────────────────────────────

def compute_ase(f):
    """
    Acoustic Shadowing Energy — Eq. (7).

    shadow(x, y) = (1 / (nz - x)) * sum_{i=x}^{nz} f(i, y)

    At each depth z, average all intensities from z to the bottom.
    Pixels just above bone see low tail mean -> complement high -> ASE high.
    Pixels below bone include the bright surface -> complement low -> suppressed.

    Args:
        f: 2D array (X x Z), values in [0, 1]. Z = depth/scanline axis.

    Returns:
        ASE: 2D array same shape, values in [0, 1].
    """
    nx, nz = f.shape

    # Sum from current depth z to bottom for each scanline
    total = f.sum(axis=1, keepdims=True)       # (nx, 1)
    cumsum_above = np.cumsum(f, axis=1)        # sum f[0..z]
    sum_from_here = total - cumsum_above + f   # sum f[z..nz-1]

    remaining = np.arange(nz, 0, -1, dtype=np.float64)  # nz, nz-1, ..., 1
    shadow = sum_from_here / remaining[np.newaxis, :]

    s_min, s_max = shadow.min(), shadow.max()
    shadow_norm = (shadow - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(shadow)

    ASE = (1.0 - shadow_norm) ** 2
    return ASE.astype(np.float32)


# ─────────────────────── Nonlinear contrast stretch — Eq. (8) ────────────────

def contrast_stretch(f, ASE):
    """
    Nonlinear piecewise contrast stretch of f * ASE — Eq. (8).

    mu(x,y) = 2 * [f*ASE]^2            if ASE <= 0.5
              1 - 2 * [1 - f*ASE]^2    if ASE >  0.5

    Args:
        f:   2D image [0, 1]
        ASE: 2D ASE image [0, 1]

    Returns:
        product: f * ASE  (before stretch, Fig 5b)
        mu:      contrast-stretched image [0, 1]
    """
    product = f * ASE
    mu = np.where(
        ASE <= 0.5,
        2.0 * product ** 2,
        1.0 - 2.0 * (1.0 - product) ** 2
    )
    return product.astype(np.float32), mu.astype(np.float32)


# ─────────────────────── Mean intensity per blob — Eq. (9) ───────────────────

def mean_intensity_per_blob(binary, mu):
    """
    MI(Bn) = (1 / Area(Bn)) * sum_{(p,q) in Bn} mu(p, q)  — Eq. (9)

    Args:
        binary: 2D uint8 binarized PSRE image
        mu:     2D float32 contrast-stretched image

    Returns:
        labeled:  2D label map
        mi_map:   2D float32 — each blob pixel filled with its MI value
        mi_dict:  dict {blob_label: MI_value}
    """
    labeled, n_blobs = label(binary)
    mi_map  = np.zeros_like(mu)
    mi_dict = {}

    for blob_id in range(1, n_blobs + 1):
        mask = labeled == blob_id
        area = mask.sum()
        if area == 0:
            continue
        mi = float(mu[mask].sum() / area)
        mi_map[mask]       = mi
        mi_dict[blob_id]   = mi

    return labeled, mi_map, mi_dict


# ─────────────────────── SRRM — Eq. (10) ─────────────────────────────────────

def compute_srrm_signal(f, blob_mask, centroid_z):
    """
    Extract the Shadow Region Row Means (SRRM) signal for one blob.

    Region: from centroid_z to bottom of image, laterally constrained
    to [x_min, x_max] of the blob.

    Args:
        f:          2D image (X x Z)
        blob_mask:  2D bool mask of this blob
        centroid_z: depth (Z) index of intensity-weighted centroid

    Returns:
        signal: 1D row means from centroid_z to bottom
        depth:  normalised depth [0, 1], same length
    """
    nx, nz = f.shape

    x_coords = np.where(blob_mask.any(axis=1))[0]
    if len(x_coords) == 0:
        return np.array([]), np.array([])

    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    z_start = int(np.clip(centroid_z, 0, nz - 1))

    region = f[x_min:x_max + 1, z_start:]  # (width x remaining_depth)
    if region.shape[1] == 0:
        return np.array([]), np.array([])

    signal = region.mean(axis=0)            # mean over X at each remaining Z
    depth  = np.linspace(0, 1, len(signal))
    return signal, depth


def fit_exponential_decay(signal, depth):
    """
    Fit y = a * exp(-b * x) to SRRM signal — Eq. (10).
    Uses Levenberg-Marquardt (method='lm').

    Returns:
        ssr:    minimised sum of squared residuals
        params: (a, b) or None if fit failed
    """
    if len(signal) < 4:
        return np.inf, None

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    try:
        p0   = [max(float(signal[0]), 1e-6), 1.0]
        popt, _ = curve_fit(exp_decay, depth, signal, p0=p0,
                            method='lm', maxfev=2000)
        ssr = float(np.sum((signal - exp_decay(depth, *popt)) ** 2))
        return ssr, popt
    except RuntimeError:
        return np.inf, None


def compute_srrm_per_blob(f, labeled, n_blobs, mu):
    """
    Compute SRRM SSR feature for every blob.

    Args:
        f:        2D US image (X x Z) [0, 1]
        labeled:  2D label map from mean_intensity_per_blob
        n_blobs:  number of blobs
        mu:       2D contrast-stretched image (for centroid weighting)

    Returns:
        srrm_map:  2D float32 — each blob pixel filled with its SSR
        srrm_dict: dict {blob_label: SSR_value}
    """
    srrm_map  = np.zeros(f.shape, dtype=np.float32)
    srrm_dict = {}

    for blob_id in range(1, n_blobs + 1):
        mask = labeled == blob_id
        if mask.sum() == 0:
            continue

        # Intensity-weighted centroid in Z
        z_coords = np.where(mask)[1]
        weights  = mu[mask]
        w_sum    = weights.sum()
        centroid_z = float((z_coords * weights).sum() / w_sum) if w_sum > 0 else float(z_coords.mean())

        signal, depth = compute_srrm_signal(f, mask, centroid_z)
        ssr, _        = fit_exponential_decay(signal, depth)

        srrm_map[mask]     = ssr
        srrm_dict[blob_id] = ssr

    return srrm_map, srrm_dict


# ─────────────────────────── Per-slice wrapper ───────────────────────────────

def compute_features_slice(f_slice, binary_slice):
    """
    Full MI + SRRM pipeline for one 2D slice.

    Returns:
        ase:       ASE image
        f_ase:     raw f * ASE product (Fig 5b)
        mu:        contrast-stretched image
        mi_map:    per-blob MI map
        mi_dict:   {blob_label: MI_value}
        srrm_map:  per-blob SSR map
        srrm_dict: {blob_label: SSR_value}
    """
    ase              = compute_ase(f_slice)
    f_ase, mu        = contrast_stretch(f_slice, ase)
    labeled, mi_map, mi_dict = mean_intensity_per_blob(binary_slice, mu)
    n_blobs          = len(mi_dict)
    srrm_map, srrm_dict = compute_srrm_per_blob(f_slice, labeled, n_blobs, mu)
    return ase, f_ase, mu, mi_map, mi_dict, srrm_map, srrm_dict


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    us_path     = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_preprocessed.nrrd"
    binary_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_preprocessed_psre_binary.nrrd"

    print("Loading volumes ...")
    us_data,     us_header = nrrd.read(us_path)
    binary_data, _         = nrrd.read(binary_path)

    print(f"  US shape:     {us_data.shape}  dtype: {us_data.dtype}")
    print(f"  Binary shape: {binary_data.shape}  dtype: {binary_data.dtype}")
    assert us_data.shape == binary_data.shape, "Shape mismatch between US and binary volumes"

    if us_data.ndim == 2:
        f      = us_data.astype(np.float64)
        binary = (binary_data > 0).astype(np.uint8)
        ase, f_ase, mu, mi_map, mi_dict, srrm_map, srrm_dict = compute_features_slice(f, binary)
        print(f"  Blobs found: {len(mi_dict)}")
        for k in sorted(mi_dict.keys()):
            print(f"    blob {k:3d}: MI = {mi_dict[k]:.4f}  SSR = {srrm_dict[k]:.6f}")
        ase_vol, f_ase_vol, mu_vol, mi_vol, srrm_vol = ase, f_ase, mu, mi_map, srrm_map

    elif us_data.ndim == 3:
        n_slices = us_data.shape[1]
        print(f"3D volume — computing MI + SRRM slice by slice along axis 1 ({n_slices} slices) ...")
        ase_s, f_ase_s, mu_s, mi_s, srrm_s = [], [], [], [], []

        for i in range(n_slices):
            if i % 10 == 0:
                print(f"  slice {i}/{n_slices}")
            f      = us_data[:, i, :].astype(np.float64)
            binary = (binary_data[:, i, :] > 0).astype(np.uint8)
            ase, f_ase, mu, mi_map, _, srrm_map, _ = compute_features_slice(f, binary)
            ase_s.append(ase)
            f_ase_s.append(f_ase)
            mu_s.append(mu)
            mi_s.append(mi_map)
            srrm_s.append(srrm_map)

        ase_vol   = np.stack(ase_s,   axis=1)
        f_ase_vol = np.stack(f_ase_s, axis=1)
        mu_vol    = np.stack(mu_s,    axis=1)
        mi_vol    = np.stack(mi_s,    axis=1)
        srrm_vol  = np.stack(srrm_s,  axis=1)
    else:
        raise ValueError(f"Unsupported ndim: {us_data.ndim}")

    out_ase   = us_path.replace(".nrrd", "_ase.nrrd")
    out_f_ase = us_path.replace(".nrrd", "_f_ase.nrrd")
    out_mu    = us_path.replace(".nrrd", "_enhanced.nrrd")
    out_mi    = us_path.replace(".nrrd", "_mi_map.nrrd")
    out_srrm  = us_path.replace(".nrrd", "_srrm_map.nrrd")

    nrrd.write(out_ase,   ase_vol.astype(np.float32),   us_header)
    nrrd.write(out_f_ase, f_ase_vol.astype(np.float32), us_header)
    nrrd.write(out_mu,    mu_vol.astype(np.float32),     us_header)
    nrrd.write(out_mi,    mi_vol.astype(np.float32),     us_header)
    nrrd.write(out_srrm,  srrm_vol.astype(np.float32),  us_header)

    print(f"Saved ASE:       {out_ase}")
    print(f"Saved f*ASE:     {out_f_ase}")
    print(f"Saved mu:        {out_mu}")
    print(f"Saved MI map:    {out_mi}")
    print(f"Saved SRRM map:  {out_srrm}")


if __name__ == "__main__":
    main()