"""
Phase-Symmetry Ridge Enhancement (PSRE) for spinal ultrasound volumes.
Implements Shajudeen & Righetti (Med. Phys. 2017), Section 2.B.
"""

import numpy as np
import nrrd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from skimage.transform import radon
from skimage.morphology import remove_small_objects


# ─────────────────────────── Log-Gabor filter ────────────────────────────────

def log_gabor_2d(shape, scale, orientation_deg, kappa_ratio=0.33, sigma_phi_deg=50.0):
    rows, cols = shape
    omega0 = 1.0 / scale
    phi0 = np.deg2rad(orientation_deg)
    sigma_phi = np.deg2rad(sigma_phi_deg)
    log_kappa = np.log(kappa_ratio)

    u = fftfreq(cols)
    v = fftfreq(rows)
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U**2 + V**2)
    radius[0, 0] = 1.0
    angle = np.arctan2(V, U)

    radial = np.exp(-(np.log(radius / omega0))**2 / (2.0 * log_kappa**2))
    radial[0, 0] = 0.0
    d_angle = np.arctan2(np.sin(angle - phi0), np.cos(angle - phi0))
    angular = np.exp(-d_angle**2 / (2.0 * sigma_phi**2))
    return radial * angular


def apply_log_gabor(image_fft, G):
    L = ifft2(image_fft * G)
    return np.real(L), np.imag(L)


# ─────────────────────── Hessian ridge strength ──────────────────────────────

def hessian_ridge_strength(image, scale):
    """
    Scale-normalised Hessian eigenvalue difference.
    Peaks at scale s matching the bone ridge width.
    """
    sigma = scale / (2 * np.sqrt(2))
    Lxx = sigma**2 * gaussian_filter(image, sigma=sigma, order=[2, 0])
    Lyy = sigma**2 * gaussian_filter(image, sigma=sigma, order=[0, 2])
    Lxy = sigma**2 * gaussian_filter(image, sigma=sigma, order=[1, 1])

    trace = Lxx + Lyy
    disc = np.sqrt((Lxx - Lyy)**2 + 4 * Lxy**2)
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)
    Ac = np.maximum(np.abs(l1) - np.abs(l2), 0)
    return Ac


# ─────────────────────────── Scale selection ─────────────────────────────────

def select_scale(image, scale_min=3, scale_max=20, n_scales=10):
    """
    Eq. (5): s* = argmax_s  sum_{x,y} A_gamma(s, x, y)
    Masked to US cone interior only (excludes black border).
    """
    scales = np.unique(np.linspace(scale_min, scale_max, n_scales).astype(int))
    us_mask = image > 0.01
    best_scale, best_score = scales[0], -np.inf
    for s in scales:
        Ac = hessian_ridge_strength(image, s)
        score = float(Ac[us_mask].sum())
        if score > best_score:
            best_score = score
            best_scale = s
    return int(best_scale)


# ─────────────────────────── Orientation selection ───────────────────────────

def select_orientations(image, scale, n_orientations=3):
    """
    Eq. (6): phi_i = argmax_phi  R_phi[ A_gamma(s*, x, y) ]
    Masked to US cone interior only.
    """
    Ac = hessian_ridge_strength(image, scale)
    Ac = Ac * (image > 0.01)  # mask black border
    theta = np.arange(0, 180, 1)
    sinogram = radon(Ac, theta=theta, circle=False)
    radon_sums = sinogram.sum(axis=0)

    sorted_idx = np.argsort(radon_sums)[::-1]
    orientations = []
    for idx in sorted_idx:
        angle = float(theta[idx])
        if all(abs(angle - a) >= 20 for a in orientations):
            orientations.append(angle)
        if len(orientations) == n_orientations:
            break
    while len(orientations) < n_orientations:
        orientations.append(orientations[-1] + 60.0)
    return orientations


# ─────────────────────────── Noise threshold T ───────────────────────────────

def rayleigh_threshold(image_fft, shape, scale, orientations,
                       kappa_ratio, sigma_phi_deg, n_sigma=1.5):
    G0 = log_gabor_2d(shape, scale, orientations[0], kappa_ratio, sigma_phi_deg)
    even0, odd0 = apply_log_gabor(image_fft, G0)
    amp = np.sqrt(even0**2 + odd0**2)
    return amp.mean() + n_sigma * amp.std()


# ─────────────────────────── PSRE (Eq. 4) ────────────────────────────────────

def compute_psre(image, scale_min=3, scale_max=20, n_scales=10,
                 n_orientations=3, kappa_ratio=0.33, sigma_phi_deg=50.0,
                 eps=1e-8, min_blob_size=50, n_sigma=1.5, thresh_pct=20):
    """
    Full PSRE for a single 2D image.

    Returns:
        psre:        continuous phase-symmetry map (float32)
        binary:      binarized + morphologically opened map (uint8)
        s_opt:       selected scale
        orientations: selected orientations (degrees)
        T:           noise threshold
    """
    shape = image.shape
    image_fft = fft2(image)

    # 1. Select optimal scale (Hessian, masked to US cone)
    s_opt = select_scale(image, scale_min, scale_max, n_scales)

    # 2. Select orientations (Radon of Hessian ridge, masked)
    orientations = select_orientations(image, s_opt, n_orientations)

    # 3. Scale triplet around s_opt
    scales = [max(scale_min, s_opt - 3), s_opt, s_opt + 3]
    scales = list(dict.fromkeys(scales))  # deduplicate

    # 4. Noise threshold from smallest scale
    T = rayleigh_threshold(image_fft, shape, min(scales), orientations,
                           kappa_ratio, sigma_phi_deg, n_sigma)

    # 5. Accumulate PS (Eq. 4)
    PS = np.zeros(shape)
    for s in scales:
        for phi in orientations:
            G = log_gabor_2d(shape, s, phi, kappa_ratio, sigma_phi_deg)
            even, odd = apply_log_gabor(image_fft, G)
            amp = np.sqrt(even**2 + odd**2) + eps
            PS += np.maximum((np.abs(even) - np.abs(odd) - T) / amp, 0.0)
    PS /= (len(scales) * n_orientations)

    # 6. Binarize: percentile of nonzero values (robust to skewed PS histograms)
    nonzero = PS[PS > 0]
    thresh = float(np.percentile(nonzero, thresh_pct)) if len(nonzero) > 100 else PS.mean()
    binary = remove_small_objects((PS > thresh), min_size=min_blob_size)

    return PS.astype(np.float32), binary.astype(np.uint8), s_opt, orientations, T


# ─────────────────────────── Debug visualization ─────────────────────────────

def debug_slice(image, ps, binary, slice_idx, s_opt, orientations, T):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray', origin='upper', aspect='auto')
    axes[0].set_title(f'Input slice {slice_idx}')
    axes[1].imshow(ps, cmap='hot', origin='upper', aspect='auto')
    axes[1].set_title(f'PSRE  (s*={s_opt}, T={T:.3f})\norientations={[f"{o:.0f}" for o in orientations]}')
    axes[2].imshow(binary, cmap='gray', origin='upper', aspect='auto')
    axes[2].set_title(f'Binary  ({binary.sum()} foreground px)')
    plt.tight_layout()
    plt.savefig('/tmp/psre_debug.png', dpi=150)
    plt.show()
    print(f"  PS stats — min: {ps.min():.4f}, max: {ps.max():.4f}, "
          f"mean: {ps.mean():.4f}, std: {ps.std():.4f}")
    print(f"  Nonzero px in PS: {(ps > 0).sum()}")
    print(f"  Foreground px in binary: {binary.sum()}")


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_preprocessed.nrrd"

    print(f"Loading {path} ...")
    data, header = nrrd.read(path)
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}")

    if data.ndim == 2:
        ps, binary, s_opt, orientations, T = compute_psre(data.astype(np.float64))
        debug_slice(data, ps, binary, 0, s_opt, orientations, T)
        psre_vol, binary_vol = ps, binary

    elif data.ndim == 3:
        n_slices = data.shape[1]
        print(f"3D volume — PSRE slice by slice along axis 1 ({n_slices} slices) ...")

        # Debug middle slice first
        mid = n_slices // 2
        sl_mid = data[:, mid, :].astype(np.float64)
        ps_mid, bn_mid, s_opt, orientations, T = compute_psre(sl_mid)
        print(f"\n--- Debug: middle slice (idx={mid}) ---")
        debug_slice(sl_mid, ps_mid, bn_mid, mid, s_opt, orientations, T)
        print("---------------------------------------\n")

        psre_slices, binary_slices = [], []
        for i in range(n_slices):
            if i % 10 == 0:
                print(f"  slice {i}/{n_slices}")
            sl = data[:, i, :].astype(np.float64)
            ps, bn, _, _, _ = compute_psre(sl)
            psre_slices.append(ps)
            binary_slices.append(bn)

        psre_vol   = np.stack(psre_slices,   axis=1)
        binary_vol = np.stack(binary_slices, axis=1)
    else:
        raise ValueError(f"Unsupported ndim: {data.ndim}")

    out_psre = path.replace(".nrrd", "_psre.nrrd")
    out_bin  = path.replace(".nrrd", "_psre_binary.nrrd")
    nrrd.write(out_psre,  psre_vol,   header)
    nrrd.write(out_bin,   binary_vol, header)
    print(f"Saved PSRE:   {out_psre}")
    print(f"Saved binary: {out_bin}")


if __name__ == "__main__":
    main()