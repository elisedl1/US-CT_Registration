"""
PSRE diagnostic script with correct scale-normalised Hessian ridge detection.

Usage:
    python psre_debug.py
    python psre_debug.py --slice 40
    python psre_debug.py --slice 40 --scale_min 3 --scale_max 30
"""

import argparse
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from skimage.transform import radon
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu


PATH = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_preprocessed.nrrd"


# ──────────────────────────────────────────────────────────────────────────────

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


def hessian_ridge_strength(image, scale):
    """
    Gamma-normalised Hessian eigenvalue difference — Eq. (5) in paper.
    Uses scale-normalised second derivatives: s^2 * d^2I/dx^2 etc.
    A_gamma = lambda1 - lambda2  where |lambda1| >= |lambda2|
    This peaks at scale s matching the ridge width.
    """
    # Scale-normalised second derivatives via Gaussian
    # s^2 * G_xx, s^2 * G_xy, s^2 * G_yy
    sigma = scale / (2 * np.sqrt(2))  # convert pixel scale to Gaussian sigma

    # Smooth first, then take finite differences for second derivatives
    smoothed = gaussian_filter(image, sigma=sigma)

    # Second derivatives (scale-normalised by sigma^2)
    Lxx = sigma**2 * (gaussian_filter(image, sigma=sigma, order=[2, 0]))
    Lyy = sigma**2 * (gaussian_filter(image, sigma=sigma, order=[0, 2]))
    Lxy = sigma**2 * (gaussian_filter(image, sigma=sigma, order=[1, 1]))

    # Hessian eigenvalues at each pixel
    # lambda = 0.5 * ((Lxx+Lyy) +/- sqrt((Lxx-Lyy)^2 + 4*Lxy^2))
    trace = Lxx + Lyy
    disc = np.sqrt((Lxx - Lyy)**2 + 4 * Lxy**2)
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)

    # Ridge strength: eigenvalue difference (positive where ridge-like)
    # Keep only where both eigenvalues are negative (dark ridge on bright background)
    # or use abs difference which is orientation-invariant
    Ac = np.abs(l1) - np.abs(l2)
    Ac = np.maximum(Ac, 0)   # only keep ridge-like responses
    return Ac


def select_scale(image, scale_min, scale_max, n_scales):
    scales = np.unique(np.linspace(scale_min, scale_max, n_scales).astype(int))
    # Mask out black border (US fan background) — only score inside the US cone
    us_mask = image > 0.01
    scores = []
    for s in scales:
        Ac = hessian_ridge_strength(image, s)
        scores.append(float(Ac[us_mask].sum()))  # sum only inside US cone
    best_scale = int(scales[np.argmax(scores)])
    return best_scale, scales, scores


def select_orientations(image, scale, n_orientations=3):
    Ac = hessian_ridge_strength(image, scale)
    # Zero out black border before Radon transform
    Ac = Ac * (image > 0.01)
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


def rayleigh_threshold(image_fft, shape, scale, orientations, kappa_ratio, sigma_phi_deg, n_sigma):
    G0 = log_gabor_2d(shape, scale, orientations[0], kappa_ratio, sigma_phi_deg)
    even0, odd0 = apply_log_gabor(image_fft, G0)
    amp = np.sqrt(even0**2 + odd0**2)
    return amp.mean() + n_sigma * amp.std()


def compute_psre_full(image, scales, orientations, kappa_ratio=0.33,
                      sigma_phi_deg=50.0, eps=1e-8, min_blob_size=50, n_sigma=3.0, thresh_pct=40):
    shape = image.shape
    image_fft = fft2(image)

    T = rayleigh_threshold(image_fft, shape, min(scales), orientations,
                           kappa_ratio, sigma_phi_deg, n_sigma)

    PS = np.zeros(shape)
    for s in scales:
        for phi in orientations:
            G = log_gabor_2d(shape, s, phi, kappa_ratio, sigma_phi_deg)
            even, odd = apply_log_gabor(image_fft, G)
            amp = np.sqrt(even**2 + odd**2) + eps
            PS += np.maximum((np.abs(even) - np.abs(odd) - T) / amp, 0.0)
    PS /= (len(scales) * len(orientations))

    nonzero = PS[PS > 0]
    # Use percentile of nonzero values — Otsu overshoots on skewed PS histograms
    thresh = float(np.percentile(nonzero, thresh_pct)) if len(nonzero) > 100 else PS.mean()
    binary = remove_small_objects((PS > thresh), min_size=min_blob_size)
    return PS.astype(np.float32), binary.astype(np.uint8), T, thresh


# ──────────────────────────────────────── Diagnostic plots ───────────────────

def plot_scale_search(scales, scores, best_scale):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(scales, scores, 'o-')
    ax.axvline(best_scale, color='r', linestyle='--', label=f's* = {best_scale}')
    ax.set_xlabel('Scale (pixels)')
    ax.set_ylabel('Sum of Hessian ridge strength')
    ax.set_title('Scale search — Hessian eigenvalue difference (Eq. 5)')
    ax.legend()
    plt.tight_layout()


def plot_orientation_search(orientations_angles, radon_sums, theta, best_scale):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(theta, radon_sums)
    for o in orientations_angles:
        ax.axvline(o, color='r', linestyle='--', alpha=0.7, label=f'{o:.0f}°')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Radon projection sum')
    ax.set_title(f'Orientation search at s={best_scale} — Eq. (6)')
    ax.legend()
    plt.tight_layout()


def plot_per_scale_response(image, scales):
    n = len(scales)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]
    for i, s in enumerate(scales):
        Ac = hessian_ridge_strength(image, s)
        axes[0, i].imshow(image, cmap='gray', origin='upper', aspect='auto')
        axes[0, i].set_title(f'Input (s={s})')
        Ac_masked = Ac * (image > 0.01)  # mask black border
        axes[1, i].imshow(Ac_masked, cmap='hot', origin='upper', aspect='auto')
        axes[1, i].set_title(f'Hessian ridge strength s={s}')
    plt.suptitle('Per-scale Hessian ridge response — bone ridges should be brightest blobs')
    plt.tight_layout()


def plot_psre_result(image, ps, binary, T, otsu_thresh, scales, orientations,
                     kappa_ratio, sigma_phi_deg, n_sigma):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray', origin='upper', aspect='auto')
    axes[0].set_title('Input slice')

    im = axes[1].imshow(ps, cmap='hot', origin='upper', aspect='auto')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title(f'PSRE map\nT={T:.4f} (n_sigma={n_sigma})')

    axes[2].hist(ps[ps > 0].ravel(), bins=100, color='steelblue')
    axes[2].axvline(otsu_thresh, color='r', linestyle='--', label=f'Otsu={otsu_thresh:.4f}')
    axes[2].set_title('PS histogram (nonzero)')
    axes[2].set_xlabel('PS value')
    axes[2].legend()

    axes[3].imshow(binary, cmap='gray', origin='upper', aspect='auto')
    axes[3].set_title(f'Binary ({binary.sum()} px)\ns={scales}, φ={[f"{o:.0f}" for o in orientations]}\n'
                      f'κ={kappa_ratio}, σ_φ={sigma_phi_deg}°')
    plt.tight_layout()
    plt.savefig('/tmp/psre_result.png', dpi=150)

    print(f"\nPS stats: min={ps.min():.4f}  max={ps.max():.4f}  "
          f"mean={ps.mean():.4f}  std={ps.std():.4f}")
    print(f"T (noise threshold): {T:.4f}")
    print(f"Otsu threshold:      {otsu_thresh:.4f}")
    print(f"Foreground px:       {binary.sum()}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',      type=int,   default=None)
    parser.add_argument('--scale_min',  type=int,   default=3)
    parser.add_argument('--scale_max',  type=int,   default=20)
    parser.add_argument('--n_scales',   type=int,   default=10)
    parser.add_argument('--kappa',      type=float, default=0.33)
    parser.add_argument('--sigma_phi',  type=float, default=50.0)
    parser.add_argument('--n_sigma',    type=float, default=1.5)
    parser.add_argument('--min_blob',   type=int,   default=50)
    parser.add_argument('--thresh_pct', type=float, default=20,
                        help='Percentile of nonzero PS values for binarization threshold (default 40)')
    args = parser.parse_args()

    data, header = nrrd.read(PATH)
    print(f"Shape: {data.shape}  dtype: {data.dtype}")

    if data.ndim == 3:
        idx = args.slice if args.slice is not None else data.shape[1] // 2
        image = data[:, idx, :].astype(np.float64)
        print(f"Using slice {idx} along axis 1")
    else:
        image = data.astype(np.float64)

    print(f"Image stats: min={image.min():.4f}  max={image.max():.4f}  "
          f"mean={image.mean():.4f}  std={image.std():.4f}")

    # 1. Scale search
    print("\n--- Scale search (Hessian eigenvalue difference) ---")
    best_scale, all_scales, scores = select_scale(
        image, args.scale_min, args.scale_max, args.n_scales)
    print(f"  s* = {best_scale}  (scores: {dict(zip(all_scales, [f'{s:.1f}' for s in scores]))})")
    plot_scale_search(all_scales, scores, best_scale)

    # 2. Orientation search
    print("--- Orientation search ---")
    Ac = hessian_ridge_strength(image, best_scale)
    theta = np.arange(0, 180, 1)
    sinogram = radon(Ac, theta=theta, circle=False)
    radon_sums = sinogram.sum(axis=0)
    orientations = select_orientations(image, best_scale)
    print(f"  orientations = {[f'{o:.1f}' for o in orientations]}")
    plot_orientation_search(orientations, radon_sums, theta, best_scale)

    # 3. Per-scale Hessian response
    scales = [max(args.scale_min, best_scale - 3), best_scale, best_scale + 3]
    scales = list(dict.fromkeys(scales))  # deduplicate
    plot_per_scale_response(image, scales)

    # 4. Full PSRE
    print("--- Computing PSRE ---")
    ps, binary, T, otsu_thresh = compute_psre_full(
        image, scales, orientations,
        kappa_ratio=args.kappa, sigma_phi_deg=args.sigma_phi,
        min_blob_size=args.min_blob, n_sigma=args.n_sigma, thresh_pct=args.thresh_pct)
    plot_psre_result(image, ps, binary, T, otsu_thresh,
                     scales, orientations, args.kappa, args.sigma_phi, args.n_sigma)

    plt.show(block=True)


if __name__ == "__main__":
    main()
# patch: this is appended — see replacement below