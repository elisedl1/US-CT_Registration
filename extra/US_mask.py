"""
make_us_mask.py

Creates a binary mask for a US NRRD volume by thresholding out zero-valued voxels.
Output mask: 1 where voxel > 0, 0 elsewhere.

Usage:
    python make_us_mask.py
    python make_us_mask.py --input /path/to/volume.nrrd --output /path/to/mask.nrrd
"""

import argparse
import numpy as np
import nrrd

DEFAULT_INPUT        = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal.nrrd"
DEFAULT_MASK_OUTPUT  = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_mask.nrrd"
DEFAULT_MASKED_OUTPUT= "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_masked.nrrd"


def make_mask(input_path: str, mask_path: str, masked_path: str) -> None:
    print(f"Loading volume: {input_path}")
    data, header = nrrd.read(input_path)

    print(f"  Shape  : {data.shape}")
    print(f"  Dtype  : {data.dtype}")
    print(f"  Min/Max: {data.min()} / {data.max()}")

    # binary mask: 1 wherever voxel is non-zero
    mask = (data != 0).astype(np.uint8)

    n_total   = mask.size
    n_nonzero = int(mask.sum())
    print(f"  Non-zero voxels: {n_nonzero} / {n_total}  ({100*n_nonzero/n_total:.1f}%)")

    # save binary mask 
    mask_header = header.copy()
    mask_header["type"] = "unsigned char"
    mask_header.setdefault("encoding", "gzip")

    print(f"Saving mask        : {mask_path}")
    nrrd.write(mask_path, mask, mask_header)

    # Save masked volume
    masked = data * mask.astype(data.dtype)

    masked_header = header.copy()
    masked_header.setdefault("encoding", "gzip")

    print(f"Saving masked volume: {masked_path}")
    nrrd.write(masked_path, masked, masked_header)

    print("Done.")


def make_mask_array(input_path: str) -> np.ndarray:
    """Return binary mask (uint8) without writing any files."""
    data, _ = nrrd.read(input_path)
    return (data != 0).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a binary mask and masked volume from a US NRRD file.")
    parser.add_argument("--input",  "-i", default=DEFAULT_INPUT,
                        help="Path to input NRRD volume")
    parser.add_argument("--mask",   "-m", default=DEFAULT_MASK_OUTPUT,
                        help="Path for output binary mask NRRD (suffix _mask)")
    parser.add_argument("--output", "-o", default=DEFAULT_MASKED_OUTPUT,
                        help="Path for output masked volume NRRD (suffix _masked)")
    args = parser.parse_args()
    make_mask(args.input, args.mask, args.output)