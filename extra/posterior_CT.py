import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from CT_axis import compute_ct_axes
from centroid import compute_centroid

def resample_to_reference(image, reference_image):
    """
    Resample an image to match the size, spacing, and orientation of a reference image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    
    resampled = resampler.Execute(image)
    return resampled


def extract_posterior_surface(mask_file, ct_file_for_axes, reference_file=None):
    """
    Extract the posterior surface of a segmentation mask along the AP axis.
    Keeps only the voxels closest to the posterior (top of AP axis).
    """

    mask = sitk.ReadImage(mask_file)
    
    if reference_file is not None:
        print(f"Resampling mask to match reference: {reference_file}")
        reference = sitk.ReadImage(reference_file)
        mask = resample_to_reference(mask, reference)
        print(f"Resampled to size: {mask.GetSize()}")
    
    mask_array = sitk.GetArrayFromImage(mask)
    
    # anatomical axes from CT
    LM_axis, AP_axis, SI_axis = compute_ct_axes(ct_file_for_axes)
    
    spacing = np.array(mask.GetSpacing())
    origin = np.array(mask.GetOrigin())
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    
    surface_array = np.zeros_like(mask_array)
    
    # for each voxel in the mask, compute its physical position
    # and project it onto the AP axis
    nonzero_indices = np.argwhere(mask_array > 0)
    
    if len(nonzero_indices) == 0:
        print("Warning: Empty mask!")
        return sitk.GetImageFromArray(surface_array)
    
    # group voxels by their LM and SI coordinates
    # to find the most posterior voxel in each "column"
    voxel_groups = {}
    
    for idx in nonzero_indices:
        z, y, x = idx
        index_vec = np.array([x, y, z])
        physical_pos = origin + direction @ (index_vec * spacing)
        
        # project onto LM and SI axes to create bins
        lm_coord = np.dot(physical_pos, LM_axis)
        si_coord = np.dot(physical_pos, SI_axis)
        ap_coord = np.dot(physical_pos, AP_axis)
        
        # round to create bins (group nearby voxels)
        lm_bin = round(lm_coord / spacing[0])
        si_bin = round(si_coord / spacing[2])
        
        key = (lm_bin, si_bin)
        
        if key not in voxel_groups:
            voxel_groups[key] = []
        
        voxel_groups[key].append((z, y, x, ap_coord))
    
    # keep only the voxel(s) with maximum AP coordinate
    for key, voxels in voxel_groups.items():
        # find maximum AP coordinate in this group
        max_ap = max(v[3] for v in voxels)
        
        # Keep voxels at or very close to the maximum
        threshold = spacing.min() * 0.5  
        for z, y, x, ap_coord in voxels:
            if abs(ap_coord - max_ap) < threshold:
                surface_array[z, y, x] = 1
    
    surface_image = sitk.GetImageFromArray(surface_array)
    surface_image.CopyInformation(mask)
    
    return surface_image


def plot_comparison(mask_file, surface_image, reference_file=None, slice_idx=None):
    """
    Plot a comparison of the original mask and the extracted surface.
    """
    # load original mask
    mask = sitk.ReadImage(mask_file)
    
    # resample to referance
    if reference_file is not None:
        reference = sitk.ReadImage(reference_file)
        mask = resample_to_reference(mask, reference)
    
    mask_array = sitk.GetArrayFromImage(mask)
    surface_array = sitk.GetArrayFromImage(surface_image)
    
    # if no slice specified, find a slice with data
    if slice_idx is None:
        # find slices with non-zero voxels
        nonzero_slices = [i for i in range(mask_array.shape[0]) if np.sum(mask_array[i, :, :]) > 0]
        if len(nonzero_slices) == 0:
            print("Warning: No slices with data found!")
            slice_idx = mask_array.shape[0] // 2
        else:
            # use the middle slice that has data
            slice_idx = nonzero_slices[len(nonzero_slices) // 2]
            print(f"Using slice {slice_idx} (has {np.sum(mask_array[slice_idx, :, :])} voxels)")
    

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # plot original mask
    axes[0].imshow(mask_array[slice_idx, :, :], cmap='gray')
    axes[0].set_title(f'Original Mask (Slice {slice_idx})')
    axes[0].axis('off')
    
    # plot surface
    axes[1].imshow(surface_array[slice_idx, :, :], cmap='gray')
    axes[1].set_title(f'Posterior Surface (Slice {slice_idx})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # statistics
    original_voxels = np.sum(mask_array > 0)
    surface_voxels = np.sum(surface_array > 0)
    print(f"\nOriginal mask voxels: {original_voxels}")
    print(f"Surface voxels: {surface_voxels}")
    print(f"Reduction: {100 * (1 - surface_voxels/original_voxels):.1f}%")



if __name__ == "__main__":
    # File paths
    mask_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L4/CT_L4.nrrd"
    ct_file_for_axes = mask_file  # using same file to get axis directions
    reference_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/open/L4/CT_L4_bt.nrrd"
    
    print("Extracting posterior surface...")
    surface = extract_posterior_surface(mask_file, ct_file_for_axes, reference_file)
    
    # save the result
    output_file = mask_file.replace('.nrrd', '_posterior_surface.nrrd')
    sitk.WriteImage(surface, output_file)
    print(f"Saved posterior surface to: {output_file}")
    
    # plot comparison
    # plot_comparison(mask_file, surface, reference_file)