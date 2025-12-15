import numpy as np
import nrrd
from numpy.fft import fftn, ifftn

def compute_monogenic_phase_amplitude(nrrd_path):
    # Load 3D image
    img, header = nrrd.read(nrrd_path)
    img = img.astype(np.float32)

    # Frequency grid
    Nx, Ny, Nz = img.shape
    fx = np.fft.fftfreq(Nx)
    fy = np.fft.fftfreq(Ny)
    fz = np.fft.fftfreq(Nz)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
    F_mag = np.sqrt(FX**2 + FY**2 + FZ**2)
    F_mag[F_mag == 0] = 1e-12

    # Fourier transform
    F = fftn(img)

    # Riesz filters
    R1 = -1j * (FX / F_mag)
    R2 = -1j * (FY / F_mag)
    R3 = -1j * (FZ / F_mag)

    # Apply Riesz transform
    Rx = np.real(ifftn(F * R1))
    Ry = np.real(ifftn(F * R2))
    Rz = np.real(ifftn(F * R3))

    # Monogenic amplitude
    A = np.sqrt(img**2 + Rx**2 + Ry**2 + Rz**2)

    # Phase amplitude (amplitude of the phase)
    A_phase = np.sqrt(Rx**2 + Ry**2 + Rz**2)

    # Local phase (for reference)
    phi = np.arctan2(A_phase, img + 1e-12)

    return A_phase, phi, A, header



input_path = "/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/sim_co_registered/US.nrrd"
A_phase, phi, A, header = compute_monogenic_phase_amplitude(input_path)

# Save amplitude as a new NRRD
nrrd.write("/Users/elisedonszelmann-lund/Masters_Utils/Rivas_Data/CaninePhantom/sim_co_registered/US_monogenic_amp.nrrd", A, header)