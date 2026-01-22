import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter

def simulate_us_artifacts(
    sitk_image,
    depth_axis=0,
    reflector_thresh=None,
    shadow_alpha=0.12,
    shadow_mode="attenuate",
    dropout_strength=0.9,
    reflection_enabled=True,
    reflection_offset=10,
    reflection_amplitude=0.6,
    reflection_lateral_sigma=2,
    reflection_count=1,
    speckle_noise_std=0.0
):
    arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    if depth_axis != 0:
        arr = np.moveaxis(arr, depth_axis, 0)
    Z, Y, X = arr.shape
    if reflector_thresh is None:
        reflector_thresh = arr.mean() + 0.5 * arr.std()
    bright_mask = arr >= reflector_thresh
    any_bright = bright_mask.any(axis=0)
    first_echo = -1 * np.ones((Y, X), dtype=np.int32)
    if any_bright.any():
        first_indices = bright_mask.argmax(axis=0)
        first_echo[any_bright] = first_indices[any_bright]
    depth_coords = np.arange(Z).reshape(Z, 1, 1)
    first_echo_b = first_echo[np.newaxis, :, :]
    distance_below = depth_coords - first_echo_b
    valid_echo_mask = (first_echo_b >= 0)
    below_mask = (distance_below > 0) & valid_echo_mask
    arr_shadowed = arr.copy()
    if shadow_mode == "attenuate":
        attenuation = np.exp(-shadow_alpha * distance_below)
        attenuation[~below_mask] = 1.0
        arr_shadowed = arr_shadowed * attenuation
    elif shadow_mode == "dropout":
        arr_shadowed[below_mask] *= (1.0 - dropout_strength)
    if speckle_noise_std > 0.0:
        noise = np.random.normal(loc=1.0, scale=speckle_noise_std, size=arr_shadowed.shape)
        arr_shadowed = arr_shadowed * noise
    arr_reflected = arr_shadowed.copy()
    if reflection_enabled:
        reflector_image = np.zeros((Y, X), dtype=np.float32)
        valid_cols = first_echo >= 0
        yy, xx = np.nonzero(valid_cols)
        reflector_image[yy, xx] = arr[first_echo[yy, xx], yy, xx]
        if reflection_lateral_sigma > 0:
            reflector_image = gaussian_filter(reflector_image, sigma=reflection_lateral_sigma)
        for i in range(reflection_count):
            offset = reflection_offset + i * (reflection_offset // max(1, reflection_count))
            ghost_depth = first_echo - offset
            ghost_mask_cols = (ghost_depth >= 0) & valid_cols
            ghost_vol = np.zeros_like(arr_reflected)
            ys, xs = np.nonzero(ghost_mask_cols)
            if ys.size > 0:
                zs = ghost_depth[ghost_mask_cols]
                ghost_vol[zs, ys, xs] = reflector_image[ys, xs] * reflection_amplitude * (1.0 / (1 + i*0.5))
                ghost_vol = gaussian_filter(ghost_vol, sigma=(1.0, reflection_lateral_sigma, reflection_lateral_sigma))
                arr_reflected = np.clip(arr_reflected + ghost_vol, a_min=0, a_max=None)
    if depth_axis != 0:
        arr_out = np.moveaxis(arr_reflected, 0, depth_axis)
    else:
        arr_out = arr_reflected
    out_sitk = sitk.GetImageFromArray(arr_out.astype(np.float32))
    out_sitk.SetSpacing(sitk_image.GetSpacing())
    out_sitk.SetOrigin(sitk_image.GetOrigin())
    out_sitk.SetDirection(sitk_image.GetDirection())
    return out_sitk

if __name__ == "__main__":
    input_nrrd  = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L2/US_weight_L2.nrrd"
    output_nrrd = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L2/US_weight_L2_dropoutref_cal.nrrd"
    print("Reading:", input_nrrd)
    img = sitk.ReadImage(input_nrrd)
    out = simulate_us_artifacts(
        img,
        depth_axis=0,
        reflector_thresh=None,
        shadow_alpha=0.20,
        shadow_mode="attenuate",
        dropout_strength=0.95,
        reflection_enabled=True,
        reflection_offset=8,
        reflection_amplitude=0.7,
        reflection_lateral_sigma=1.5,
        reflection_count=2,
        speckle_noise_std=0.02
    )
    sitk.WriteImage(out, output_nrrd)
    print("Saved:", output_nrrd)

    arr_in  = sitk.GetArrayFromImage(img)
    arr_out = sitk.GetArrayFromImage(out)
    mid_slice = arr_in.shape[0] // 2
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Mid Slice (z={mid_slice})")
    plt.imshow(arr_in[mid_slice], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f"Shadow + Reflection Artifacts (z={mid_slice})")
    plt.imshow(arr_out[mid_slice], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
