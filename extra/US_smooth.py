"""
NRRD Smoothing Script with Brightness Enhancement
"""

import numpy as np
import nrrd
from scipy.ndimage import gaussian_filter, white_tophat, grey_dilation
from skimage import exposure
import os
import sys

def enhance_bright_regions(data, method='gamma', **kwargs):
    
    #normalize to 0-1 range for processing
    data_min, data_max = data.min(), data.max()
    normalized = (data - data_min) / (data_max - data_min)
    
    if method == 'gamma':
        gamma = kwargs.get('gamma', 0.5)
        print(f"  Using gamma correction with gamma={gamma}")
        enhanced = np.power(normalized, gamma)
        
    elif method == 'tophat':
        #  top-hat transform highlights bright features
        size = kwargs.get('size', 5)
        print(f"  Using white top-hat transform with size={size}")
        enhanced = np.zeros_like(normalized)
        for i in range(data.shape[0]):
            enhanced[i] = white_tophat(normalized[i], size=size)
        enhanced = normalized + enhanced
        enhanced = np.clip(enhanced, 0, 1)
        
    elif method == 'clahe':
        # CLAHE enhances local contrast
        print(f"  Using CLAHE (Contrast Limited Adaptive Histogram Equalization)")
        enhanced = np.zeros_like(normalized)
        for i in range(data.shape[0]):
            enhanced[i] = exposure.equalize_adapthist(normalized[i], clip_limit=0.03)
    
    elif method == 'sigmoid':
        # Sigmoid contrast enhancement
        gain = kwargs.get('gain', 10)
        cutoff = kwargs.get('cutoff', 0.5)
        print(f"  Using sigmoid enhancement: gain={gain}, cutoff={cutoff}")
        enhanced = exposure.adjust_sigmoid(normalized, cutoff=cutoff, gain=gain)
    
    else:
        print(f"  Unknown method '{method}', using original data")
        enhanced = normalized
    
    enhanced_data = enhanced * (data_max - data_min) + data_min
    
    return enhanced_data

def smooth_and_enhance_nrrd(input_path, sigma=1.0, enhance_method='gamma', **enhance_kwargs):

    print(f"Loading NRRD file from: {input_path}")
    
    data, header = nrrd.read(input_path)
    
    # gaussian smoothing first
    print(f"\nApplying Gaussian smoothing with sigma={sigma}...")
    smoothed_data = gaussian_filter(data, sigma=sigma)
    
    # enhance bright regions
    print(f"Enhancing bright regions using '{enhance_method}' method...")
    enhanced_data = enhance_bright_regions(smoothed_data, method=enhance_method, **enhance_kwargs)
    
    enhanced_data = enhanced_data.astype(data.dtype)

    # output files    
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(directory, f"{name}_smoothed_enhanced{ext}")
    
    print(f"\nSaving enhanced image to: {output_path}")
    nrrd.write(output_path, enhanced_data, header)
    
    print("Done!")
    return output_path

if __name__ == "__main__":
    input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa6/Cases/US_complete_cal.nrrd"
    
    # Smoothing parameters
    # sigma = 1.0 is mild smoothing
    # sigma = 2.0 is moderate smoothing
    # sigma = 3.0+ is strong smoothing
    sigma = 1.0
    
    # Enhancement method options:
    # 'gamma' - good for emphasizing bright regions (recommended: gamma=0.5)
    # 'tophat' - highlights bright features (recommended: size=5)
    # 'clahe' - enhances local contrast
    # 'sigmoid' - sigmoid contrast enhancement
    enhance_method = 'tophat'
    
    # parameters 
    enhance_params = {
        'gamma': 0.5,  # Lower = more emphasis on bright regions (try 0.3-0.7)
        # 'size': 5,   # For tophat method
        # 'gain': 10,  # For sigmoid method
        # 'cutoff': 0.5,  # For sigmoid method
    }
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        sigma = float(sys.argv[2])
    if len(sys.argv) > 3:
        enhance_method = sys.argv[3]
    if len(sys.argv) > 4 and enhance_method == 'gamma':
        enhance_params['gamma'] = float(sys.argv[4])
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    output_path = smooth_and_enhance_nrrd(input_path, sigma=sigma, 
                                          enhance_method=enhance_method, 
                                          **enhance_params)
    print(f"Enhanced file saved as: {output_path}")