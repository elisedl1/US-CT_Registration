import nrrd
import numpy as np
from utils.helpers import preprocess_US

# Input/output paths
input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal.nrrd"
output_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra_axial_cal/US_complete_cal_TEST.nrrd"

# Call the function
ase_vol, header = preprocess_US(
    input_path,
    method='tophat',
    sigma=1.0,
    size=5
)

# Save the output volume
nrrd.write(output_path, ase_vol, header)
print(f"Saved preprocessed volume to: {output_path}")
print(f"Output shape: {ase_vol.shape}, dtype: {ase_vol.dtype}")