"""
Log-Gabor bone enhancement following the OCMI pipeline:
  - Eq. (1): Log-Gabor filter in frequency domain
  - Eq. (2): Even/odd responses via inverse FFT
  - Eq. (3): Amplitude A_so and phase angle φ_so
  - Eq. (4): Phase deviation Δφ_so
  - Eq. (5): Enhanced image I'(x,y)

6 orientations × 4 scales = 24 filters applied per 2D slice.
The 3D volume is processed slice-by-slice along the axial (z) axis.
"""

import os
import numpy as np
import nrrd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal/aniso/aniso_preprocessed.nrrd"
output_dir = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertevra_axial_two_cal"

# ── Log-Gabor parameters (matching paper) ─────────────────────────────────────
N_ORIENTATIONS = 6          # θ_0 values evenly spaced over [0, π)
N_SCALES       = 4          # ω_0 values (center frequencies)
KAPPA          = 0.65       # bandwidth scaling factor κ  (paper default ~0.65)
EPSILON        = 1e-6       # ε_0 in eq. (5) — prevents division by zero

# Center frequencies for each scale (cycles/pixel).
# Adjust to match the spatial resolution of your data.
OMEGA_0_LIST = [0.05, 0.1, 0.2, 0.4]   # ω_0 per scale


# ── Helpers ────────────────────────────────────────────────────────────────────

def log_gabor_2d(rows, cols, omega_0, theta_0, kappa, sigma_theta):
    """
    Build a 2-D Log-Gabor filter in the frequency domain — eq. (1):

        G(ω, θ) = exp( -[(log(ω/ω_0))² / (2·(log(κ/ω_0))²)
                         + (θ - θ_0)² / (2·σ_θ²)] )

    Returns a real-valued array of shape (rows, cols).
    """
    u = np.fft.fftfreq(cols)
    v = np.fft.fftfreq(rows)
    U, V = np.meshgrid(u, v)

    radius = np.sqrt(U**2 + V**2)
    radius[0, 0] = 1.0                # avoid log(0) at DC

    angle = np.arctan2(V, U)

    # Radial log-Gaussian component
    log_rad_term = (np.log(radius / omega_0))**2 / (2 * (np.log(kappa / omega_0))**2)
    radial = np.exp(-log_rad_term)
    radial[0, 0] = 0.0                # zero DC

    # Angular component — wrap difference to [-π/2, π/2]
    d_theta = angle - theta_0
    d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
    angular = np.exp(-(d_theta**2) / (2 * sigma_theta**2))

    return radial * angular


def apply_log_gabor_slice(image_2d, omega_0, theta_0, kappa, sigma_theta):
    """
    Apply one Log-Gabor filter to a single 2-D slice.
    Returns (e_so, o_so) — even (real) and odd (imaginary) responses.
    """
    rows, cols = image_2d.shape
    F = np.fft.fft2(image_2d)
    G = log_gabor_2d(rows, cols, omega_0, theta_0, kappa, sigma_theta)
    response = np.fft.ifft2(F * G)
    return np.real(response), np.imag(response)


# ── Load volume ────────────────────────────────────────────────────────────────
print(f"Loading: {input_path}")
volume, header = nrrd.read(input_path)
print(f"  Shape: {volume.shape}  dtype: {volume.dtype}")

volume = volume.astype(np.float32)
nz = volume.shape[2]

# σ_θ: angular bandwidth so adjacent orientations overlap at half-max
sigma_theta = np.pi / N_ORIENTATIONS
thetas = [o * np.pi / N_ORIENTATIONS for o in range(N_ORIENTATIONS)]

# ── Process slice-by-slice ─────────────────────────────────────────────────────
enhanced = np.zeros_like(volume)

for z in range(nz):
    if z % 20 == 0:
        print(f"  Slice {z}/{nz} ...")

    slc = volume[:, :, z].astype(np.float64)

    numerator   = np.zeros_like(slc)
    denominator = np.zeros_like(slc)

    for omega_0 in OMEGA_0_LIST:
        for theta_0 in thetas:

            # Eq. (2): even/odd responses
            e_so, o_so = apply_log_gabor_slice(slc, omega_0, theta_0,
                                               KAPPA, sigma_theta)

            # Eq. (3): amplitude and phase angle
            A_so   = np.sqrt(e_so**2 + o_so**2)
            phi_so = np.arctan2(o_so, e_so)

            # Eq. (4): phase deviation (mean phase over the slice at this orientation)
            phi_mean  = np.mean(phi_so)
            delta_phi = (np.cos(phi_so - phi_mean)
                         - np.abs(np.sin(phi_so - phi_mean)))

            # Eq. (5): accumulate
            numerator   += A_so * delta_phi
            denominator += A_so

    # Eq. (5): normalised enhanced slice
    enhanced[:, :, z] = np.clip(numerator / (denominator + EPSILON), 0, None).astype(np.float32)

# ── Save ───────────────────────────────────────────────────────────────────────
Path(output_dir).mkdir(parents=True, exist_ok=True)

out_header = {k: header[k] for k in ("space", "space directions", "space origin") if k in header}
out_header["type"]      = "float"
out_header["dimension"] = enhanced.ndim
out_header["sizes"]     = list(enhanced.shape)

out_path = os.path.join(output_dir, "log_gabor_enhanced.nrrd")
nrrd.write(out_path, enhanced, out_header)
print(f"\nSaved: {out_path}")
print("Done.")