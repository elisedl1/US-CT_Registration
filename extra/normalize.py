import nrrd 
import numpy as np 

# normalize iUS intensity volume
iUS_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3/US_L3.nrrd"
iUS_data, header = nrrd.read(iUS_path)

# Normalize to [0, 1]
iUS_norm = (iUS_data - np.min(iUS_data)) / (np.max(iUS_data) - np.min(iUS_data))

# Save normalized image
output_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3/US_L3_nonweight_norm.nrrd"
nrrd.write(output_path, iUS_norm, header)