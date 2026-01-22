import os
import time
import numpy as np
import torch
import SimpleITK as sitk
import cma
from glob import glob
import multiprocessing as mp 
from functools import partial
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
from collections import defaultdict

from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix, compute_inter_vertebral_displacement_penalty, compute_ivd_collision_loss, compute_facet_collision_loss
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid
from extra.CT_axis import compute_ct_axes
from extra.IVD_points import compute_adjacent_vertebra_pairings



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
                       moving_parsers, fixed_parser,
                       case_centroids, orig_dists, case_axes, pairings, facet_pairings, 
                       case_names, device='cuda', profile=False):


    total_sim = 0.0
    transforms_params = []
    moved_centroids = []
    transforms_list = []

    for k in range(K):

        params = torch.tensor(flat_params[6*k:6*(k+1)], dtype=torch.float32, device=device)
        transforms_params.append(params.cpu().numpy())

        # get transformation       
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k:6*(k+1)].tolist())
        tx_inv = tx.GetInverse() # inverse transformm CT -> US
        transforms_list.append(tx_inv)
    

        # sampled CT surface points - x from CT
        sampled_positions = sampled_positions_list[k].to(device=device, dtype=torch.float64) 

        # transform CT -> US       
        M = sitk_euler_to_matrix(tx_inv) 
        M_torch = torch.from_numpy(M).to(device=device, dtype=torch.float64) 
        
        N = sampled_positions.shape[0]
        pts_h = torch.cat([sampled_positions, torch.ones((N, 1), device=device, dtype=sampled_positions.dtype)], dim=1)
        moved_positions = (pts_h @ M_torch.T)[:, :3]

        # sample US (fixed) intensities
        moving_vals = fixed_parser.sample_at_physical_points_gpu(moved_positions) # US
        moving_intensities = moving_vals.float() # US intensities 

        # mean iUS intensity metric
        sim = torch.mean(moving_intensities)
        total_sim += sim
        
        # centroid transform
        ct_centroid = torch.tensor(case_centroids[k], device=device, dtype=torch.float32)
        moved_centroid = torch.tensor(tx_inv.TransformPoint(ct_centroid.cpu().numpy().tolist()), device=device)
        moved_centroids.append(moved_centroid.cpu().numpy())

    mean_sim = total_sim / float(K)


    # INTER-VERTEBRAL DISPLACEMENT PENALTY
    lambda_axes = 0.1
    axes_margins = { # mm, degree values
        'LM': 2.0,
        'AP': 2.0,
        'SI': 5.0,
        'LM_rot' : np.deg2rad(15), # bending (forward, backwards)
        'AP_rot' : np.deg2rad(6), # side bending
        'SI_rot' : np.deg2rad(2) # twisting
    }

    axes_penalty = compute_inter_vertebral_displacement_penalty(
        moved_centroids, case_centroids, case_axes, transforms_list, axes_margins
    )



    # IVD POINT PAIR PENALTY
    lambda_ivd = 0.000
    ivd_loss, ivd_metrics = compute_ivd_collision_loss(pairings, transforms_list, case_names)
    

    # FACET POINT PAIR PENALTY
    lambda_facet = 0.0 # was 0.001
    facet_loss, facet_metrics = compute_facet_collision_loss(facet_pairings, transforms_list, case_names)


    # TOTAL LOSS
    sim_weight = 1.0
    total_loss = (
        sim_weight * -float(mean_sim) + 
        (lambda_axes * axes_penalty) +
        (lambda_ivd * ivd_loss) +
        (lambda_facet * facet_loss)
    )


    return (
        total_loss,
        float(mean_sim * (sim_weight)),
        float(axes_penalty * lambda_axes),
        float(ivd_loss * lambda_ivd),
        float(facet_loss * lambda_facet),
        ivd_metrics
    )



if __name__ == "__main__":
    
    # suppress PyVista cleanup warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*NoneType.*check_attribute.*')
    
    # SETTINGS
    # form
    mesh_dir = '/usr/local/data/elise/pig_data/pig2/Registration/cropped/intra1'
    cases_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/Cases'
    output_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/output_python_cma_group_allcases'

    USE_GLOBAL_RANDOM_PERTURBATION = False
    PERT_TRANSLATION_MM = 5.0
    PERT_ROTATION_DEG = 10.0
    PERT_SEED = 123



    

    # gather case folders (assumes L1..L4 style)
    os.makedirs(output_dir, exist_ok=True)
    case_names = sorted([
        name for name in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
    ])

    # ----------
    print("Computing nearest IVD points for adjacent vertebra...")

    suffix_ivd = "_body.vtk"
    try:
        pairings, meshes = compute_adjacent_vertebra_pairings(
        mesh_dir,
        suffix_ivd,
        n_sample=30000,
        n_pairs=100,# change number of pairs between each vertebra
        max_dist=7.0,
        seed=42
        )
        
        # (pairings contain numpy arrays, not mesh references)
        for mesh in meshes.values():
            mesh.clear_data()
        meshes.clear()
        del meshes
    except Exception as e:
        print(f"Error computing pairings: {e}")

    # ----------
    print("Computing nearest Facet Joint points for adjacent vertebra...")

    suffix_facet = "_upper.vtk"
    try:
        facet_pairings, facet_meshes = compute_adjacent_vertebra_pairings(
        mesh_dir,
        suffix_facet,
        n_sample=30000,
        n_pairs=50,# change number of pairs between each vertebra
        max_dist=4.0,
        seed=42
        )
        
        # (pairings contain numpy arrays, not mesh references)
        for mesh in facet_meshes.values():
            mesh.clear_data()
        facet_meshes.clear()
        del facet_meshes
    except Exception as e:
        print(f"Error computing facet pairings: {e}")



    # convert to np for ivd and facet
    for k in pairings:
        pairings[k]['L_i'] = np.asarray(pairings[k]['L_i'])
        pairings[k]['L_j'] = np.asarray(pairings[k]['L_j'])
    for k in facet_pairings:
        facet_pairings[k]['L_i'] = np.asarray(facet_pairings[k]['L_i'])
        facet_pairings[k]['L_j'] = np.asarray(facet_pairings[k]['L_j'])



    # ----------
    print("Group-wise registration for cases:", case_names)

    # read single US volume once
    # fixed_file = os.path.join(cases_dir, 'US_complete.nrrd') # old
    # fixed_file = os.path.join(cases_dir, 'US_complete_cal.nrrd') # NEW
    # fixed_file = os.path.join(cases_dir, 'US_full_L3_dropoutref.nrrd') # old
    fixed_file = os.path.join(cases_dir, 'US_full_L3_dropoutref_cal.nrrd') # NEW
    # fixed_file = os.path.join(cases_dir, 'US_full_L2_L3_dropoutref.nrrd')
    fixed_parser = PyNrrdParser(fixed_file) 

    # precompute per-case data
    moving_parsers = []
    sampled_positions_list = []
    centers = []
    case_landmarks = []
    case_axes = []

    for case in case_names:
        print(f"\nPreparing case {case} ...")
        case_path = os.path.join(cases_dir, case)
        moving_file = os.path.join(case_path, 'moving.nrrd') # CT for case
        
        # read in fixed and moving images
        moving_parser = PyNrrdParser(moving_file) # CT
        moving_parsers.append(moving_parser) # CT
        moving_tensor = moving_parser.get_tensor(False) # CT

        # sample CT (moving) points at posterior surface
        mask = moving_tensor > 0 # ct image
        ct_mask_indices = torch.stack(torch.where(mask), dim=-1)
        num_vox = ct_mask_indices.shape[0]
        samples_count = min(5000, num_vox)
        if samples_count == 0:
            raise RuntimeError(f"No positive voxels found in fixed file {fixed_file} for case {case}")

        samples = torch.randint(num_vox, (samples_count,))
        sampled_indices = ct_mask_indices[samples]
        sampled_positions = moving_parser.compute_positions(sampled_indices)
        sampled_positions_list.append(sampled_positions)

    
        # compute center of moving CT images (for transformation)
        moving_img = sitk.ReadImage(moving_file)
        center = np.array(
            moving_img.TransformContinuousIndexToPhysicalPoint(
                np.array(moving_img.GetSize()) / 2.0
            )
        )
        centers.append(center)


        # LANDMARKS
        # form
        target_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/US_{case}_landmarks.mrk.json" # matches fixed US (intra)
        source_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case}_landmarks_intra.mrk.json" # matches moving CT
        
        # local
        # target_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/US_{case}_landmarks.mrk.json" # matches fixed US (intra)
        # source_file = f"/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case}_landmarks_intra.mrk.json" # matches moving CT
        

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


        # compute CT anatomical axes
        ct_seg_file = os.path.join(cases_dir, case, f'CT_{case}.nrrd')
        if not os.path.exists(ct_seg_file):
            raise FileNotFoundError(f"CT segmentation not found for case {case}")
        
        LM_axis, AP_axis, SI_axis = compute_ct_axes(ct_seg_file)
        case_axes.append((LM_axis, AP_axis, SI_axis))


    # compute centroids of fixed CT once
    case_centroids = []
    for case in case_names:
        seg_file = os.path.join(cases_dir, case, f'CT_{case}.nrrd')

        if not os.path.exists(seg_file):
            raise FileNotFoundError(f"fixed.nrrd not found for case {case}")

        c = compute_centroid(seg_file)
        case_centroids.append(np.array(c))


    # compute original inter-centroid distances once
    orig_dists = []
    for k in range(len(case_centroids) - 1):
        d = np.linalg.norm(case_centroids[k] - case_centroids[k + 1])
        orig_dists.append(float(d))
    orig_dists = np.array(orig_dists) 


    # K = number of structures
    K = len(case_names)
    print(f"\nPrepared {K} cases -> optimizing {6*K} parameters jointly.")

    # GPU check
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    # move data to GPU if available
    print("Preparing tensors on GPU...")
    sampled_positions_list_gpu = [torch.from_numpy(pos.astype(np.float32)).cuda() for pos in sampled_positions_list]



    # random perturbation
    # rng = np.random.default_rng(PERT_SEED)
    rng = np.random.default_rng()


    if USE_GLOBAL_RANDOM_PERTURBATION:
        # rotations (radians)
        rot = np.deg2rad(
            rng.uniform(-PERT_ROTATION_DEG, PERT_ROTATION_DEG, size=3)
        )

        # translations (mm)
        trans = rng.uniform(
            -PERT_TRANSLATION_MM, PERT_TRANSLATION_MM, size=3
        )

        # SITK Euler3D: (rx, ry, rz, tx, ty, tz)
        global_perturbation = np.concatenate([rot, trans])
    else:
        global_perturbation = np.zeros(6)

    if USE_GLOBAL_RANDOM_PERTURBATION:
        print("\nApplied GLOBAL random CT perturbation:")
        print(f"  Rotation (deg): {np.rad2deg(global_perturbation[:3])}")
        print(f"  Translation (mm): {global_perturbation[3:]}")

    
    # create partial evaluate function 
    partial_eval = partial(
        evaluate_group_gpu,
        K=K,
        centers=centers,
        sampled_positions_list=sampled_positions_list_gpu,
        moving_parsers=moving_parsers,
        fixed_parser=fixed_parser,
        case_centroids=case_centroids,
        orig_dists=orig_dists,
        case_axes=case_axes,
        pairings=pairings,
        facet_pairings = facet_pairings,
        case_names=case_names,
        device='cuda',
        profile=False
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
    # x0 = np.zeros(6 * K) 
    x0 = np.tile(global_perturbation, K)
    sigma0 = 0.5
    base_stds = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    cma_stds = base_stds * K

    popsize = 80
    parents = 20
    lower_per = [-0.4, -0.4, -0.4, -5, -5, -5]
    upper_per = [0.4, 0.4, 0.4, 5, 5, 5]
    lower = lower_per * K
    upper = upper_per * K

    # track metrics
    loss_history = []
    mean_sim_history = []
    axes_penalty_history = []
    ivd_loss_history = [] 
    facet_loss_history = []
    ivd_log = []

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
        }
    )


    if USE_GLOBAL_RANDOM_PERTURBATION:
        print("\nTRE AFTER GLOBAL PERTURBATION (before CMA)")
        tre_pert = compute_case_tre(x0)
        for case, tre in tre_pert.items():
            if tre is not None:
                print(f"{case}: TRE_perturbed = {tre:.4f} mm")



    # RUN CMA ON GPU
    start_time = time.time()
    it = 0

    while not es.stop():
        solutions = es.ask()
        values = []
            
        for sol in solutions:

            val, mean_sim, axes_pen, ivd_loss, facet_loss, ivd_metrics = partial_eval(sol)
                
            values.append(val)
            mean_sim_history.append(mean_sim)
            axes_penalty_history.append(axes_pen)
            ivd_loss_history.append(ivd_loss)
            facet_loss_history.append(facet_loss)
            loss_history.append(val)
            
            ivd_log.append({
                "eval_id": len(ivd_log),
                "total_loss": float(val),
                "mean_sim": float(mean_sim),
                "axes_penalty": float(axes_pen),
                "ivd_loss": float(ivd_loss),
                "ivd_metrics": ivd_metrics
            })

        es.tell(solutions, values)      
        it += 1


    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"\nGroup CMA finished after {it} iterations — time {mins} min {secs:.2f} sec")


    # IVD analysis 
    with open(os.path.join(output_dir, "ivd_diagnostics.json"), "w") as f:
        json.dump(ivd_log, f, indent=2)




    # save transforms per-case - ALL IN MAIN OUTPUT DIRECTORY
    best_flat = es.result.xbest
    final_transforms = []
    for k, case in enumerate(case_names):
        params = best_flat[6*k:6*(k+1)]
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(params.tolist())

        # save directly in output_dir with case name suffix
        out_name = os.path.join(output_dir, f"TransformParameters_groupwise_{case}.h5")
        sitk.WriteTransform(tx, out_name)
        print(f"Wrote transform for {case}: {out_name}")
        
        # store inverse transform for IVD spacing calculation
        tx_inv = tx.GetInverse()
        final_transforms.append(tx_inv)


    # compute and print TRE per-case
    tres = compute_case_tre(best_flat)
    for case, tre in tres.items():
        if tre is None:
            print(f"{case}: no landmarks available -> TRE not computed")
        else:
            print()
            print(f"{case}: TRE = {tre:.1f}")

    mean_sim_arr = -1 * np.array(mean_sim_history)
    axes_penalty_arr = np.array(axes_penalty_history)
    ivd_loss_arr = np.array(ivd_loss_history) 
    facet_loss_arr = np.array(facet_loss_history)
    loss_arr = np.array(loss_history)


    # LOSS PLOTTING
    plt.figure(figsize=(10, 6))
    plt.plot(np.where(mean_sim_arr != 0)[0], mean_sim_arr[mean_sim_arr != 0], 'o', label='Mean Similarity', linestyle='None')
    plt.plot(np.where(axes_penalty_arr != 0)[0], axes_penalty_arr[axes_penalty_arr != 0], 's', label='Axes Penalty', linestyle='None')
    plt.plot(np.where(ivd_loss_arr != 0)[0], ivd_loss_arr[ivd_loss_arr != 0], '^', label='IVD Spacing Loss', linestyle='None')
    plt.plot(np.where(facet_loss_arr != 0)[0], facet_loss_arr[facet_loss_arr != 0], 'v', label='Facet Loss', linestyle='None')
    plt.plot(np.where(loss_arr != 0)[0], loss_arr[loss_arr != 0], '.', label='Total Loss', linestyle='None')

    plt.xlabel('CMA Evaluation Step')
    plt.ylabel('Value')
    plt.title('CMA Optimization Metrics per Evaluation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimization_metrics.png", dpi=150)
    plt.close()

    print(f"Saved optimization_metrics.png in {output_dir}")






    # --- Log final rotation differences between adjacent vertebra ---
print("\nFinal inter-vertebral rotation differences (deg):")
rotation_log = []

for k in range(len(final_transforms) - 1):
    tx_k = final_transforms[k].GetInverse()  # get original transform
    tx_k1 = final_transforms[k + 1].GetInverse()

    rot_k = np.array(final_transforms[k].GetMatrix()).reshape(3, 3)
    rot_k1 = np.array(final_transforms[k + 1].GetMatrix()).reshape(3, 3)

    LM_k, AP_k, SI_k = case_axes[k]
    LM_k1, AP_k1, SI_k1 = case_axes[k + 1]

    # rotate axes
    LM_k_rot = rot_k @ LM_k
    AP_k_rot = rot_k @ AP_k
    SI_k_rot = rot_k @ SI_k

    LM_k1_rot = rot_k1 @ LM_k1
    AP_k1_rot = rot_k1 @ AP_k1
    SI_k1_rot = rot_k1 @ SI_k1

    # axis-angle differences
    def axis_angle(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos_theta)

    SI_rot_deg = np.rad2deg(axis_angle(SI_k_rot, SI_k1_rot))  # rotation about SI
    LM_rot_deg = np.rad2deg(axis_angle(LM_k_rot, LM_k1_rot))  # rotation about LM
    AP_rot_deg = np.rad2deg(axis_angle(AP_k_rot, AP_k1_rot))  # rotation about AP

    rotation_log.append({
        "pair": f"{case_names[k]}-{case_names[k+1]}",
        "SI_rot_deg": SI_rot_deg,
        "LM_rot_deg": LM_rot_deg,
        "AP_rot_deg": AP_rot_deg
    })

    print(f"{case_names[k]}-{case_names[k+1]} -> SI: {SI_rot_deg:.2f}, LM: {LM_rot_deg:.2f}, AP: {AP_rot_deg:.2f}")
