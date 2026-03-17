from scipy.ndimage import gaussian_filter, white_tophat
import nrrd 
import numpy as np 

def preprocess_US(input_path, method='tophat', sigma=1.0, size=5):
    
    # read in image, gaussian smooth
    data, header = nrrd.read(input_path)
    smoothed_data = gaussian_filter(data, sigma=sigma)

    if method == 'none':
        return smoothed_data, header
    
    normalized = smoothed_data
    
    # top-hat transform highlights bright features
    enhanced = np.zeros_like(normalized)
    for i in range(data.shape[0]):
        enhanced[i] = white_tophat(normalized[i], size=size)
    enhanced = normalized + enhanced
    enhanced = np.clip(enhanced, 0, 1)  
    
    return enhanced, header


if __name__ == '__main__':
    path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa6/Cases/US_complete_cal.nrrd"
    output_path = path.replace('.nrrd', '_preprocessed.nrrd')

    enhanced, header = preprocess_US(path, method='tophat', sigma=1.0, size=5)

    nrrd.write(output_path, enhanced, header)
    print(f"Saved preprocessed image to {output_path}")
