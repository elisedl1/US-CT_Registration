import os
import time
import numpy as np
import torch
import SimpleITK as sitk
import cma
import matplotlib.pyplot as plt

from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix
from utils.similarity import IntensitySimilarity



def compute_tre(flat_params, ct_lms, us_lms, center):
    """Compute Target Registration Error"""
    if ct_lms is None or us_lms is None:
        return None
    
    # create transformation from params
    params = np.array(flat_params, dtype=float)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center.tolist())
    tx.SetParameters(params.tolist())
    tx_inv = tx.GetInverse()
    
    # compute moved landmarks
    moved = torch.tensor([tx_inv.TransformPoint(p.tolist()) for p in ct_lms.cpu().numpy()])
    diff = moved - us_lms

    return float(torch.mean(torch.linalg.norm(diff, dim=1)).item())



def evaluate(flat_params, center, sampled_positions, moving_parser):
    
    # get transformation
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center.tolist())
    tx.SetParameters(flat_params.tolist())
    tx_inv = tx.GetInverse()  # Inverse transform: CT -> US
    
    # transform CT -> US
    M = sitk_euler_to_matrix(tx_inv)
    M_torch = torch.from_numpy(M).to(dtype=sampled_positions.dtype)
    N = sampled_positions.shape[0]
    pts_h = torch.cat([sampled_positions, torch.ones((N, 1), dtype=sampled_positions.dtype)], dim=1)
    moved_positions = (pts_h @ M_torch.T)[:, :3]
    
    # sample US (fixed) intensities at transformed positions
    moving_vals = fixed_parser.sample_at_physical_points_gpu(moved_positions)
    moving_intensities = moving_vals.float()
    
    # mean iUS intensity metric
    sim = torch.mean(moving_intensities)
    mean_sim = float(sim)
    total_loss = -mean_sim
    
    return total_loss, mean_sim


if __name__ == "__main__":
    
    case_name = 'L2'
    cases_dir = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases'
    output_dir = f'/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/output_python_cma/{case_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    case_path = os.path.join(cases_dir, case_name)
    fixed_file = os.path.join(case_path, 'fixed.nrrd')  # US
    moving_file = os.path.join(case_path, 'moving.nrrd')  # CT
    
    print(f"Processing case {case_name}...")
    
    # IMAGES
    moving_parser = PyNrrdParser(moving_file)  # CT
    moving_tensor = moving_parser.get_tensor(False)
    fixed_parser = PyNrrdParser(fixed_file)  # US
    fixed_tensor = fixed_parser.get_tensor(False)
    
    # sample CT (moving) points at posterior surface
    mask = moving_tensor > 0  # CT image
    ct_mask_indices = torch.stack(torch.where(mask), dim=-1)
    num_vox = ct_mask_indices.shape[0]
    samples_count = min(10000, num_vox)
    if samples_count == 0:
        raise RuntimeError(f"No positive voxels found in moving file {moving_file}")
    
    samples = torch.randint(num_vox, (samples_count,))
    sampled_indices = ct_mask_indices[samples]
    sampled_positions = moving_parser.compute_positions(sampled_indices)
    
    # compute center of fixed image
    fixed_img = sitk.ReadImage(fixed_file)
    center = np.array(
        fixed_img.TransformContinuousIndexToPhysicalPoint(
            np.array(fixed_img.GetSize()) / 2.0
        )
    )
    
    # LANDMARKS
    target_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/US_{case_name}_landmarks_intra.mrk.json"  # US (fixed)
    source_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case_name}_landmarks.mrk.json"  # CT (moving)
    
    try:
        fixed_lm_parser = SlicerJsonTagParser(target_file)  # Fixed US
        moving_lm_parser = SlicerJsonTagParser(source_file)  # Moving CT
        ct_lms = moving_lm_parser.extract_landmarks()
        us_lms = fixed_lm_parser.extract_landmarks()
        print("Landmarks found")
    except Exception as e:
        ct_lms = None
        us_lms = None
        print(f"No landmarks available: {e}")
    
    # initial TRE
    print("\nTRE BEFORE OPTIMIZATION")
    tre_before = compute_tre(np.zeros(6), ct_lms, us_lms, center)
    if tre_before is None:
        print("No landmarks available -> TRE not computed")
    else:
        print(f"TRE_before = {tre_before:.4f} mm\n")
    

    sampled_positions_tensor = torch.from_numpy(sampled_positions.astype(np.float32))
    
    # CMA initialization
    x0 = np.zeros(6)
    sigma0 = 0.5
    cma_stds = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    
    popsize = 80
    parents = 20
    lower = [-0.4, -0.4, -0.4, -5, -5, -5]
    upper = [0.4, 0.4, 0.4, 5, 5, 5]
    
    # CMA loop
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        options={
            'CMA_stds': cma_stds,
            'popsize': popsize,
            'CMA_mu': parents,
            'bounds': [lower, upper],
            'verb_disp': 1,
            'maxiter': 80,
            'tolfun': 1e-5,
            'seed': 772512
        }
    )
    
    
    # RUN CMA
    start_time = time.time()
    it = 0
    
    start_time = time.time()
    while not es.stop():
        solutions = es.ask()
        values = []
        for sol in solutions:
            val, mean_sim = evaluate(sol, center, sampled_positions_tensor, moving_parser)
            values.append(val)
        es.tell(solutions, values)
        it += 1
    
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"\nCMA finished after {it} iterations — time {mins} min {secs:.2f} sec")
    
    best_flat = es.result.xbest
    print("Best params:", best_flat)
    
    # save transform
    params = best_flat
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center.tolist())
    tx.SetParameters(params.tolist())
    out_name = os.path.join(output_dir, f"TransformParameters_{case_name}.h5")
    sitk.WriteTransform(tx, out_name)
    print(f"Wrote transform: {out_name}")
    
    # new tre
    tre_after = compute_tre(best_flat, ct_lms, us_lms, center)
    if tre_after is None:
        print("No landmarks available -> TRE not computed")
    else:
        print(f"\nTRE_after = {tre_after:.4f} mm")
    
    
    print("DONE.")