import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from extra.centroid import compute_centroid

def compute_ct_axes(ct_file):
    """
    Compute anatomical axes from CT image direction matrix.
    Returns LM (lateral-medial), AP (anterior-posterior), SI (superior-inferior) axis vectors.
    """
    ct_image = sitk.ReadImage(ct_file)
    D = np.array(ct_image.GetDirection()).reshape(3, 3)
    
    LM_axis = D[:, 0]  # Lateral-Medial
    AP_axis = D[:, 1]  # Anterior-Posterior
    SI_axis = D[:, 2]  # Superior-Inferior
    
    return LM_axis, AP_axis, SI_axis