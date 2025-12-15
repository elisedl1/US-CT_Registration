import SimpleITK as sitk
import os

# input file
input_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/intra1_seg/L1_L2_disc_mask_good.nrrd"

# read the disc mask
disc_image = sitk.ReadImage(input_file)

# fill internal holes (3d)
disc_filled = sitk.BinaryFillhole(disc_image, foregroundValue=1)

# save filled mask with _filled suffix
folder, fname = os.path.split(input_file)
name, ext = os.path.splitext(fname)
out_file = os.path.join(folder, f"{name}_filled.nrrd")

sitk.WriteImage(disc_filled, out_file)

print(f"filled disc mask saved to: {out_file}")