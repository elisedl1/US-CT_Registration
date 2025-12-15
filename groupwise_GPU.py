
import os
import time
import numpy as np
import torch
import SimpleITK as sitk
import cma
from glob import glob
import multiprocessing as mp 
from functools import partial

from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid


def compute_case_tre(flat_params):
    tre_results = {}
    for k, case in enumerate(case_names):
        ct_lms, us_lms = case_landmarks[k] # CT moving, US fixed
        if ct_lms is None or us_lms is None:
            tre_results[case] = None
            continue
        params = np.array(flat_params[6*k:6*(k+1)], dtype=float)
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(params.tolist())
        tx_inv = tx.GetInverse()
        moved = torch.tensor([tx_inv.TransformPoint(p.tolist()) for p in ct_lms.cpu().numpy()])
        diff = moved - us_lms
        tre_results[case] = float(torch.mean(torch.linalg.norm(diff, dim=1)).item())
    return tre_results


def evaluate_group_gpu(flat_params, K, centers, sampled_positions_list,
                       moving_parsers, fixed_parsers,
                       case_centroids, orig_dists, device='cuda'):

    total_sim = 0.0
    transforms_params = []
    moved_centroids = []

    for k in range(K):
        params = torch.tensor(flat_params[6*k:6*(k+1)], dtype=torch.float32, device=device)
        transforms_params.append(params.cpu().numpy())

        # get transformation
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k:6*(k+1)].tolist())
        tx_inv = tx.GetInverse() # inverse transformm CT -> US

        # sampled CT surface points - x from CT
        sampled_positions = sampled_positions_list[k].to(device=device, dtype=torch.float64) 

        # transform CT -> US
        M = sitk_euler_to_matrix(tx_inv) 
        M_torch = torch.from_numpy(M).to(device=device, dtype=torch.float64) 
        N = sampled_positions.shape[0]
        pts_h = torch.cat([sampled_positions, torch.ones((N, 1), device=device, dtype=sampled_positions.dtype)], dim=1)
        moved_positions = (pts_h @ M_torch.T)[:, :3]

        # sample US (fixed) intensities
        moving_vals = fixed_parsers[k].sample_at_physical_points_gpu(moved_positions) # US parser
        moving_intensities = moving_vals.float() # US intensities 
        # fixed_intensities = fixed_intensities_list[k]

        # mean iUS intensity metric
        sim = torch.mean(moving_intensities)
        total_sim += sim

        # centroid transform
        ct_centroid = torch.tensor(case_centroids[k], device=device, dtype=torch.float32)

        # transform into fixed US space - use inverse
        moved_centroid = torch.tensor(tx_inv.TransformPoint(ct_centroid.cpu().numpy().tolist()), device=device)
        moved_centroids.append(moved_centroid.cpu().numpy())

    mean_sim = total_sim / float(K)


    # centroid distance penalty
    lambda_centroid = 0.0
    margin_mm = 4.0
    centroid_penalty = 0.0
    if K > 1:
        for k in range(K-1):
            new_dist = np.linalg.norm(moved_centroids[k] - moved_centroids[k+1])
            diff = abs(new_dist - orig_dists[k]) - margin_mm
            if diff > 0.0:
                centroid_penalty += diff**2
        centroid_penalty /= float(K-1)




    # print(
    #     f"mean_sim: {mean_sim:.4f}, "
    #     f"adjacency_penalty: {lambda_smooth * adjacency_penalty:.4f}, "
    #     f"centroid_penalty: {lambda_centroid * centroid_penalty:.4f}, "
    #     f"collision_penalty: {collision_penalty:.4f}, "
    #     f"total_loss: {-float(mean_sim) + lambda_smooth * adjacency_penalty + lambda_centroid * centroid_penalty + collision_penalty:.4f}"
    # )


    return -float(mean_sim) + (lambda_centroid * centroid_penalty)


if __name__ == "__main__":
    
    # SETTINGS
    # form
    cases_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/Cases'
    output_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/output_python_cma_group_allcases'
    os.makedirs(output_dir, exist_ok=True)

    # Gather case folders (assumes L1..L4 style)
    case_names = sorted([
        name for name in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
    ])
    print("Group-wise registration for cases:", case_names)

    # Precompute per-case data
    moving_parsers = []
    fixed_parsers = []
    sampled_positions_list = []
    fixed_intensities_list = []
    centers = []
    case_landmarks = []
    case_output_dirs = []

    for case in case_names:
        print(f"\nPreparing case {case} ...")
        case_path = os.path.join(cases_dir, case)
        fixed_file = os.path.join(case_path, 'fixed.nrrd')
        moving_file = os.path.join(case_path, 'moving.nrrd')
        
        # read in fixed and moving images
        moving_parser = PyNrrdParser(moving_file) # CT
        moving_parsers.append(moving_parser)
        moving_tensor = moving_parser.get_tensor(False)
        fixed_parser = PyNrrdParser(fixed_file) # US
        fixed_parsers.append(fixed_parser)
        fixed_tensor = fixed_parser.get_tensor(False)

        # sample CT (moving) points at posterior surface
        mask = moving_tensor > 0 # ct image
        ct_mask_indices = torch.stack(torch.where(mask), dim=-1)
        num_vox = ct_mask_indices.shape[0]
        samples_count = min(10000, num_vox)
        if samples_count == 0:
            raise RuntimeError(f"No positive voxels found in fixed file {fixed_file} for case {case}")

        samples = torch.randint(num_vox, (samples_count,))
        sampled_indices = ct_mask_indices[samples]
        sampled_positions = moving_parser.compute_positions(sampled_indices)
        sampled_positions_list.append(sampled_positions)

        # fixed_intensities = fixed_tensor[
        #     sampled_indices[:, 0],
        #     sampled_indices[:, 1],
        #     sampled_indices[:, 2]
        # ].float()
        # fixed_intensities_list.append(fixed_intensities)

    
        # compute center of fixed image to add to transform at end
        fixed_img = sitk.ReadImage(fixed_file)
        center = np.array(
            fixed_img.TransformContinuousIndexToPhysicalPoint(
                np.array(fixed_img.GetSize()) / 2.0
            )
        )
        centers.append(center)

        # read in landmarks
        target_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/US_{case}_landmarks_intra.mrk.json" # matches fixed US (intra)
        source_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case}_landmarks.mrk.json" # matches moving CT
        
        try:
            fixed_lm_parser = SlicerJsonTagParser(target_file) # fixed US
            moving_lm_parser = SlicerJsonTagParser(source_file) # moving CT
            ct_lms = moving_lm_parser.extract_landmarks()
            us_lms = fixed_lm_parser.extract_landmarks()
            case_landmarks.append((ct_lms, us_lms))
            print("  Landmarks found for case", case)
        except Exception:
            case_landmarks.append((None, None))
            print("  No landmarks for case", case)

        case_out = os.path.join(output_dir, case)
        os.makedirs(case_out, exist_ok=True)
        case_output_dirs.append(case_out)


    # compute centroids of fixed CT once
    case_centroids = []
    for case in case_names:
        seg_file = os.path.join(cases_dir, case, f'CT_{case}.nrrd')

        if not os.path.exists(seg_file):
            raise FileNotFoundError(f"fixed.nrrd not found for case {case}")

        c = compute_centroid(seg_file)
        case_centroids.append(np.array(c))
    print("centroids: ", case_centroids)


    # compute original inter-centroid distances once
    orig_dists = []
    for k in range(len(case_centroids) - 1):
        d = np.linalg.norm(case_centroids[k] - case_centroids[k + 1])
        orig_dists.append(float(d))
    orig_dists = np.array(orig_dists)  # shape (K-1,)


    # compute CT anatomical axes 
    # TODO THIS

    # K = number of structures
    K = len(case_names)
    print(f"\nPrepared {K} cases -> optimizing {6*K} parameters jointly.")

    # GPU check
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    # move data to GPU if available
    print("Preparing tensors on GPU...")
    sampled_positions_list_gpu = [torch.from_numpy(pos.astype(np.float32)).cuda() for pos in sampled_positions_list]
    fixed_intensities_list_gpu = [x.cuda() for x in fixed_intensities_list]


    # create partial evaluate function
    partial_eval = partial(
        evaluate_group_gpu,
        K=K,
        centers=centers,
        sampled_positions_list=sampled_positions_list_gpu,
        moving_parsers=moving_parsers,
        fixed_parsers=fixed_parsers,
        case_centroids=case_centroids,
        orig_dists=orig_dists,
        device='cuda'
    )


    # print initial TRE
    print("\nTRE BEFORE OPTIMIZATION")
    tre_before = compute_case_tre(np.zeros(6 * K))
    for case, tre in tre_before.items():
        if tre is None:
            print(f"{case}: no landmarks available -> TRE not computed")
        else:
            print(f"{case}: TRE_before = {tre:.4f} mm")


    # CMA initialization
    x0 = np.zeros(6 * K) 
    sigma0 = 0.5
    base_stds = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    cma_stds = base_stds * K

    popsize = 80
    parents = 20
    lower_per = [-0.4, -0.4, -0.4, -5, -5, -5]
    upper_per = [0.4, 0.4, 0.4, 5, 5, 5]
    lower = lower_per * K
    upper = upper_per * K

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

    # RUN CMA ON GPU
    start_time = time.time()
    it = 0

    while not es.stop():
        solutions = es.ask()
        values = [partial_eval(sol) for sol in solutions] 
        es.tell(solutions, values)
        it += 1

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"\nGroup CMA finished after {it} iterations — time {mins} min {secs:.2f} sec")

    best_flat = es.result.xbest
    print("Best first-6 params (case 1):", best_flat[:6])


    # save transforms per-case
    for k, case in enumerate(case_names):
        params = best_flat[6*k:6*(k+1)]
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(params.tolist())

        # print(case, params)

        out_name = os.path.join(case_output_dirs[k], f"TransformParameters_groupwise.h5")
        sitk.WriteTransform(tx, out_name)
        print(f"Wrote transform for {case}: {out_name}")


    # compute and print TRE per-case
    tres = compute_case_tre(best_flat)
    for case, tre in tres.items():
        if tre is None:
            print(f"{case}: no landmarks available -> TRE not computed")
        else:
            print()
            print(f"{case}: TRE = {tre:.4f}")

    print("DONE.")

    '''
    Ius(x) * Ict(T(x))

    Ius(T-1(x))
    '''