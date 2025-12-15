import os
from functools import reduce
import SimpleITK as sitk
import torch
import numpy as np
import cma
import time
# from scipy.optimize import differential_evolution

from utils.file_parser import TagFileParser, SlicerJsonTagParser, PyNrrdParser

from utils.finite_differences import (
    fd_3d_intensity_at_indices
)

from utils.helpers import (
    euler_matrix
)
# from utils.logger import LogIO

from utils.similarity import IntensitySimilarity
import matplotlib.pyplot as plt



# input images
ct_file = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/fixed.nrrd' # fixed CT
# us_file = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/moving.nrrd'  # moving US
us_file = '/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/US_Vertebra/L3/US_L3_nonweight_norm.nrrd'
# landmarks
target_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/CT_L3_landmarks_intra.mrk.json" # matches fixed CT
source_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/open/CT_landmarks/CT_L3_landmarks.mrk.json" # matches moving US

# target_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/open/CT_landmarks/CT_L3_landmarks.mrk.json" # fixed CT 
# source_file = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/CT_L3_landmarks_intra.mrk.json" # moving US
moving_file_parser = SlicerJsonTagParser(source_file)
fixed_file_parser = SlicerJsonTagParser(target_file)
ct_landmarks = fixed_file_parser.extract_landmarks()
us_landmarks = moving_file_parser.extract_landmarks()
# print("ct_landmarks: ", ct_landmarks)
# print("us_landmarks: ", us_landmarks)

moving_parser = PyNrrdParser(us_file)
fixed_parser = PyNrrdParser(ct_file)
moving_tensor = moving_parser.get_tensor(True)  # US
fixed_tensor = fixed_parser.get_tensor(False)  # CT
# print(fixed_parser.size)
# print(fixed_parser.spacing)

# get samples of CT binary mask
mask = fixed_tensor > 0
fixed_mask_indices = torch.stack(torch.where(mask), dim=-1) 
num_voxels_1 = fixed_mask_indices.shape[0]
# samples_count = min(10_000, num_voxels_1)
samples_count = 10000
samples = torch.randint(num_voxels_1, (samples_count,))

masked_fix_intensities = fixed_tensor[
    fixed_mask_indices[samples][:, 0],
    fixed_mask_indices[samples][:, 1],
    fixed_mask_indices[samples][:, 2]
]

fixed_img = sitk.ReadImage(ct_file)
center = np.array(fixed_img.TransformContinuousIndexToPhysicalPoint(np.array(fixed_img.GetSize()) / 2.0))
# print("fixed_tensor type and shape: ", type(fixed_tensor), fixed_tensor.shape)
# print("moving_tensor type and shape: ", type(moving_tensor), moving_tensor.shape)
# print()

# print("fixed_mask_indices type and shape: ", type(fixed_mask_indices), fixed_mask_indices.shape)
# print("samples type and shape: ", type(samples), samples.shape, samples[0])
# print("masked_fix_intensities type and shape: ", type(masked_fix_intensities), masked_fix_intensities.shape, masked_fix_intensities[0])
# print()


## physical space with sitk transform
def validate_rigid(rigid_matrix = None, h5_file = None):

    if h5_file is not None:
        tx = sitk.ReadTransform(h5_file)
    elif rigid_matrix is not None:
        params = np.array(rigid_matrix, dtype=float)
        tx = sitk.Euler3DTransform()
        tx.SetCenter(center.tolist())
        tx.SetParameters(params.tolist())
    else:
        raise ValueError("Either rigid_matrix or h5_file must be provided.")

    # invert tx
    tx_inv = tx.GetInverse()

    moved = []
    for p in us_landmarks.cpu().numpy():
        moved.append(tx_inv.TransformPoint(p.tolist()))
    moved = torch.tensor(moved)

    diff = moved - ct_landmarks
    return torch.mean(torch.linalg.norm(diff, dim=1)).item()

# rigid_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# print(validate_rigid(rigid_params))

# compute physical positions of sampled mask voxels (CT)
sampled_indices = fixed_mask_indices[samples]
sampled_positions = fixed_parser.compute_positions(sampled_indices)
fixed_intensities = fixed_tensor[
    sampled_indices[:, 0],
    sampled_indices[:, 1],
    sampled_indices[:, 2]
].float()



def evaluate(rigid_matrix) -> float:

    params = np.array(rigid_matrix, dtype=float)

    # apply rigid transform in CT physical space
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center.tolist()) # def need this
    tx.SetParameters(params)
    moved_positions = np.array([tx.TransformPoint(tuple(p)) for p in sampled_positions])
    

    # sample US image at transformed CT points
    moving_vals_np = moving_parser.sample_at_physical_points(moved_positions)    
    moving_intensities = torch.from_numpy(moving_vals_np).float()

    f = IntensitySimilarity.compute(fixed_intensities, moving_intensities)

    return -f

# sanity check
# print(evaluate([0.0, 0.0 ,0.0, 0.0 ,0.0, 0.0 ]))
# print(evaluate([0.1, 0.0 , 0.0 , 0.0 , 0.0, 0.0 ]))


# OPTIMIZER
# initial guess for rigid parameters: rotations (rad) + translations (mm)
x0 = [0,0,0,0,0,0]  
N = 6

# scales
sigma0 = 1.0
cma_stds = [0.05, 0.05, 0.05, 2.0, 2.0, 2.0]

# population and parent size
popsize = 60
parents = 30

# bounds (rot rad / trans mm)
lower = [-0.4, -0.4, -0.4, -5, -5, -5] 
upper = [0.4, 0.4, 0.4, 5, 5, 5]

es = cma.CMAEvolutionStrategy(
    x0, sigma0,
    options={
        'CMA_stds': cma_stds,
        'popsize': popsize,
        'CMA_mu': parents,
        'bounds': [lower, upper],
        'verb_disp':1,
        'maxiter':20,
        'tolfun': 1e-5,
        'seed': 772512
    }
)
start_time = time.time()  # start timing
while not es.stop():
    solutions = es.ask()
    values = [evaluate(x) for x in solutions]
    es.tell(solutions, values)

best_params = es.result.xbest
print("Best rigid params:", best_params)
print("Landmark error:", validate_rigid(best_params))

end_time = time.time()  # end timing

# create transform with best params
fixed_img = sitk.ReadImage(ct_file)
center = np.array(fixed_img.TransformContinuousIndexToPhysicalPoint(
    np.array(fixed_img.GetSize()) / 2.0
))

tx = sitk.Euler3DTransform()
tx.SetCenter(center.tolist())
tx.SetParameters(best_params)
# print(tx)
output_h5 = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/L3/output_python_cma/TransformParameters.h5"
sitk.WriteTransform(tx, output_h5)


# compute TRE
error_h5 = validate_rigid(h5_file = output_h5)
print("Landmark error from H5:", error_h5)

# print registration time
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60
print(f"Optimization took {minutes} min {seconds:.2f} sec")