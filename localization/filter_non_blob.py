import argparse
import numpy as np
import nrrd

_BASE = (
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/"
    "Registration/US_Vertebra_axial_cal/"
)
DEFAULT_VOLUME = _BASE + "US_complete_cal.nrrd"
DEFAULT_MASK   = _BASE + "US_complete_cal_bone_candidates_filtered.nrrd"
DEFAULT_OUT    = _BASE + "US_complete_cal_blanked.nrrd"

DIMMING_FACTOR = 0.1


def run(volume_path, mask_path, out_path):
    print(f"Loading volume : {volume_path}")
    volume, header = nrrd.read(volume_path)
    print(f"  shape : {volume.shape}  dtype : {volume.dtype}")

    print(f"Loading mask   : {mask_path}")
    mask, _ = nrrd.read(mask_path)

    out = volume.astype(np.float32).copy()
    out[mask > 0] = out[mask > 0] * DIMMING_FACTOR
    n_blanked = int((mask > 0).sum())
    print(f"  blanked voxels : {n_blanked} / {volume.size}")

    out_hdr = header.copy()
    out_hdr["type"] = "float"
    out_hdr.setdefault("encoding", "gzip")

    print(f"Saving : {out_path}")
    nrrd.write(out_path, out, out_hdr)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume",      "-v", default=DEFAULT_VOLUME)
    parser.add_argument("--mask",        "-m", default=DEFAULT_MASK)
    parser.add_argument("--out",         "-o", default=DEFAULT_OUT)
    parser.add_argument("--dimming-factor", "-d", type=float, default=DIMMING_FACTOR)
    args = parser.parse_args()

    run(args.volume, args.mask, args.out)