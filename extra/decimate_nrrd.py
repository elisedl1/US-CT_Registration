import os
import SimpleITK as sitk

# ==========================
# USER SETTINGS
# ==========================
DECIMATION_FACTOR = 2  # 2 = half resolution

INPUT_FILES = [
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L1.nrrd",
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L2.nrrd",
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L3.nrrd",
    "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L4.nrrd",
]

OUTPUT_DIR = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original_sofa"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# DECIMATION FUNCTION
# ==========================
def decimate_labelmap(img, factor):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_spacing = tuple(s * factor for s in original_spacing)
    new_size = tuple(int(sz / factor) for sz in original_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img)

# ==========================
# MAIN LOOP
# ==========================
for path in INPUT_FILES:
    print(f"Decimating: {path}")

    img = sitk.ReadImage(path, sitk.sitkUInt16)
    decimated = decimate_labelmap(img, DECIMATION_FACTOR)

    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}_decimated.nrrd")

    sitk.WriteImage(decimated, out_path)
    print(f"Saved -> {out_path}")

print("All segmentations decimated and saved to original_sofa.")
