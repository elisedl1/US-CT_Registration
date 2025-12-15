import os
import time
import numpy as np
import torch
import SimpleITK as sitk
import cma

from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import euler_matrix
from utils.similarity import IntensitySimilarity

# SETTINGS
cases_dir = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases'
output_dir = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/output_python_cma'

os.makedirs(output_dir, exist_ok=True)

case_names = sorted([
    name for name in os.listdir(cases_dir)
    if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
])  # ['L1', 'L2', 'L3', 'L4']
print(case_names)

# LOOP OVER CASES
for case in case_names:
    print(f"\n=== Processing case {case} ===")
    
    case_path = os.path.join(cases_dir, case)
    ct_file = os.path.join(case_path, 'fixed.nrrd')
    us_file = os.path.join(case_path, 'moving.nrrd')
    
    # landmarks
    target_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/open/CT_landmarks/CT_{case}_landmarks.mrk.json"
    source_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case}_landmarks_intra.mrk.json"
    
    moving_file_parser = SlicerJsonTagParser(source_file)
    fixed_file_parser = SlicerJsonTagParser(target_file)
    ct_landmarks = fixed_file_parser.extract_landmarks()
    us_landmarks = moving_file_parser.extract_landmarks()
    
    moving_parser = PyNrrdParser(us_file)
    fixed_parser = PyNrrdParser(ct_file)
    
    moving_tensor = moving_parser.get_tensor(True)
    fixed_tensor = fixed_parser.get_tensor(False)
    
    # sample mask points
    mask = fixed_tensor > 0
    fixed_mask_indices = torch.stack(torch.where(mask), dim=-1)
    num_voxels_1 = fixed_mask_indices.shape[0]
    samples_count = min(10000, num_voxels_1)
    samples = torch.randint(num_voxels_1, (samples_count,))
    
    sampled_indices = fixed_mask_indices[samples]
    sampled_positions = fixed_parser.compute_positions(sampled_indices)

    # compute physical positions of sampled mask voxels (CT)
    fixed_intensities = fixed_tensor[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2]
    ].float()
    
    fixed_img = sitk.ReadImage(ct_file)
    center = np.array(fixed_img.TransformContinuousIndexToPhysicalPoint(
        np.array(fixed_img.GetSize()) / 2.0
    ))
    
    # EVALUATION FUNCTION
    def evaluate(rigid_matrix) -> float:
        params = np.array(rigid_matrix, dtype=float)
        tx = sitk.Euler3DTransform()
        tx.SetCenter(center.tolist())
        tx.SetParameters(params)
        
        moved_positions = np.array([tx.TransformPoint(tuple(p)) for p in sampled_positions])
        moving_vals_np = moving_parser.sample_at_physical_points(moved_positions)    
        moving_intensities = torch.from_numpy(moving_vals_np).float()
        
        ct_vals_np = fixed_parser.sample_at_physical_points(sampled_positions)
        ct_intensities = torch.from_numpy(ct_vals_np).float()
        
        f = IntensitySimilarity.compute(fixed_intensities, moving_intensities)
        return -f
    
    # VALIDATION
    def validate_rigid(rigid_matrix=None, h5_file=None):
        if h5_file is not None:
            tx = sitk.ReadTransform(h5_file)
        elif rigid_matrix is not None:
            params = np.array(rigid_matrix, dtype=float)
            tx = sitk.Euler3DTransform()
            tx.SetParameters(params.tolist())
        else:
            raise ValueError("Either rigid_matrix or h5_file must be provided.")
        
        moved = torch.tensor([tx.TransformPoint(p.tolist()) for p in us_landmarks.cpu().numpy()])
        diff = moved - ct_landmarks
        return torch.mean(torch.linalg.norm(diff, dim=1)).item()
    
    # OPTIMIZATION
    x0 = [0, 0, 0, 0, 0, 0]  
    sigma0 = 1.0
    cma_stds = [0.05, 0.05, 0.05, 2.0, 2.0, 2.0]
    popsize = 60
    parents = 30
    lower = [-0.4, -0.4, -0.4, -5, -5, -5] 
    upper = [0.4, 0.4, 0.4, 5, 5, 5]
    
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        options={
            'CMA_stds': cma_stds,
            'popsize': popsize,
            'CMA_mu': parents,
            'bounds': [lower, upper],
            'verb_disp': 1,
            'maxiter': 20,
            'tolfun': 1e-5,
            'seed': 772512
        }
    )
    
    start_time = time.time()
    while not es.stop():
        solutions = es.ask()
        values = [evaluate(x) for x in solutions]
        es.tell(solutions, values)
    
    best_params = es.result.xbest
    # print("Best rigid params:", best_params)
    # print("Landmark error:", validate_rigid(best_params))
    
    # SAVE TRANSFORM
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center.tolist())
    tx.SetParameters(best_params)
    
    case_output_dir = os.path.join(output_dir, case)
    os.makedirs(case_output_dir, exist_ok=True)
    output_h5 = os.path.join(case_output_dir, "TransformParameters.h5")
    sitk.WriteTransform(tx, output_h5)
    
    # TRE
    error_h5 = validate_rigid(h5_file=output_h5)
    print("Landmark error from H5:", error_h5)
    
    # time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Optimization took {minutes} min {seconds:.2f} sec")
