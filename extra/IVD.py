import os
import re
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

# USER OPTIONS
use_folder = True   # set False to process only two files

folder = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/"

# if not using folder, specify two files manually
manual_file_a = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/CT_L3_body.nrrd"
manual_file_b = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/CT_L4_body.nrrd"




def compute_disc(seg_a, seg_b, max_gap=10):
    arr_a = sitk.GetArrayFromImage(seg_a) > 0
    arr_b = sitk.GetArrayFromImage(seg_b) > 0

    dist_a = distance_transform_edt(~arr_a)
    dist_b = distance_transform_edt(~arr_b)

    disc = ((dist_a + dist_b) < max_gap).astype(np.uint8)
    disc[arr_a] = 0
    disc[arr_b] = 0

    out = sitk.GetImageFromArray(disc)
    out.CopyInformation(seg_a)
    return out


# MODE A — PROCESS ALL PAIRS IN FOLDER
if use_folder:

    all_files = os.listdir(folder)
    vertebra_files = []
    pattern = re.compile(r"ct_l(\d+)_body\.nrrd", re.IGNORECASE)

    for f in all_files:
        match = pattern.match(f)
        if match:
            level = int(match.group(1))
            vertebra_files.append((level, os.path.join(folder, f)))

    vertebra_files = [f[1] for f in sorted(vertebra_files, key=lambda x: x[0])]

    print("detected vertebra files:")
    for f in vertebra_files:
        print("  ", os.path.basename(f))

    for i in range(len(vertebra_files) - 1):
        file_a = vertebra_files[i]
        file_b = vertebra_files[i + 1]

        name_a = os.path.basename(file_a).replace(".nrrd", "")
        name_b = os.path.basename(file_b).replace(".nrrd", "")

        out_name = f"{name_a}_{name_b}_disc_mask.nrrd"
        out_path = os.path.join(folder, out_name)

        print(f"\nprocessing disc: {name_a} – {name_b}")

        seg_a = sitk.ReadImage(file_a)
        seg_b = sitk.ReadImage(file_b)
        seg_b = sitk.Resample(seg_b, seg_a)

        disc_img = compute_disc(seg_a, seg_b, max_gap=15)

        sitk.WriteImage(disc_img, out_path)
        print("saved:", out_path)

    print("\nall discs computed!")


# MODE B — PROCESS ONLY TWO FILES
else:
    file_a = manual_file_a
    file_b = manual_file_b

    name_a = os.path.basename(file_a).replace(".nrrd", "")
    name_b = os.path.basename(file_b).replace(".nrrd", "")

    out_name = f"{name_a}_{name_b}_disc_mask.nrrd"
    out_path = os.path.join(os.path.dirname(file_a), out_name)

    print(f"processing single pair: {name_a} – {name_b}")

    seg_a = sitk.ReadImage(file_a)
    seg_b = sitk.ReadImage(file_b)
    seg_b = sitk.Resample(seg_b, seg_a)

    disc_img = compute_disc(seg_a, seg_b, max_gap=15)
    sitk.WriteImage(disc_img, out_path)

    print("saved:", out_path)
