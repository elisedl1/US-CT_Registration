import numpy as np
import sys
from CT_axis import compute_ct_axes


def compute_lumbar_lordosis_angle(l1_file, l4_file):
    """
    Compute the lumbar lordosis angle (LLA) between L1 and L4 vertebrae.
    
    The LLA is defined as the angle between the AP (anterior-posterior) axes
    of L1 and L4 vertebrae, as visualized in sagittal plane.
    
    Parameters:
    -----------
    l1_file : str
        Path to the L1 CT segmentation file (.nrrd)
    l4_file : str
        Path to the L4 CT segmentation file (.nrrd)
    
    Returns:
    --------
    angle_degrees : float
        Lumbar lordosis angle in degrees
    """
    # Get anatomical axes for L1 and L4
    l1_lm, l1_ap, l1_si = compute_ct_axes(l1_file)
    l4_lm, l4_ap, l4_si = compute_ct_axes(l4_file)
    
    # Compute angle between AP axes
    # Using dot product: cos(theta) = (v1 · v2) / (|v1| * |v2|)
    dot_product = np.dot(l1_ap, l4_ap)
    
    # Normalize (should already be normalized, but for safety)
    l1_ap_norm = np.linalg.norm(l1_ap)
    l4_ap_norm = np.linalg.norm(l4_ap)
    
    cos_angle = dot_product / (l1_ap_norm * l4_ap_norm)
    
    # Clamp to valid range to avoid numerical errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute angle in radians, then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) == 3:
        # User provided two file paths
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        
        print("Computing angle between AP axes...")
        print("=" * 50)
        print(f"File 1: {file1}")
        print(f"File 2: {file2}")
        
        angle = compute_lumbar_lordosis_angle(file1, file2)
        print(f"\nAngle between AP axes: {angle:.2f}°")
        print("=" * 50)
    
    elif len(sys.argv) > 1:
        print("Usage:")
        print("  python LLA.py <file1.nrrd> <file2.nrrd>  # Compute angle between two files")
        print("  python LLA.py                             # Run default comparison")
        sys.exit(1)
    
    else:
        # Default behavior: compare original vs moving segmentations
        # Original CT segmentations
        l1_original = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L1.nrrd"
        l4_original = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/original/CT_L4.nrrd"
        
        # Moving/transformed CT segmentations
        l1_moving = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa6/Cases/all_moving/CT_L1.nrrd"
        l4_moving = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/sofa6/Cases/all_moving/CT_L4.nrrd"
        
        # Compute LLA for original segmentations
        print("Computing Lumbar Lordosis Angle (LLA)...")
        print("=" * 50)
        
        lla_original = compute_lumbar_lordosis_angle(l1_original, l4_original)
        print(f"\nOriginal CT Segmentations:")
        print(f"  LLA (L1-L4): {lla_original:.2f}°")
        
        # Compute LLA for moving/transformed segmentations
        lla_moving = compute_lumbar_lordosis_angle(l1_moving, l4_moving)
        print(f"\nMoving/Transformed CT Segmentations:")
        print(f"  LLA (L1-L4): {lla_moving:.2f}°")
        
        # Compute difference
        lla_difference = abs(lla_moving - lla_original)
        print(f"\nDifference: {lla_difference:.2f}°")
        print("=" * 50)