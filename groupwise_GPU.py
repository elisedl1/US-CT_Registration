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
from enum import Enum
# imports from my files
from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix, sigmoid_ramp, compute_inter_vertebral_displacement_penalty, compute_ivd_collision_loss, compute_facet_collision_loss
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid
from extra.CT_axis import compute_ct_axes
from extra.IVD_points import compute_adjacent_vertebra_pairings


# EXPERIMENT CONFIGURATION
class ExperimentType(Enum):
    NORMAL = "normal"
    FULL_SWEEP = "full_sweep"
    MISSING_DATA = "missing_data"
    ROBUSTNESS = "robustness"


# CHANGE THIS TO SELECT EXPERIMENT
EXPERIMENT = ExperimentType.NORMAL
SUCCESS_THRESH_MM = 2.0


def get_experiment_settings(exp_type):
    """define experiment-specific settings"""
    if exp_type == ExperimentType.NORMAL:
        return {
            "us_files": ["US_full_L3_dropoutref_cal.nrrd"], # can change this to L3
            "perturb": False,
            "n_runs": 1
        }
    
    if exp_type == ExperimentType.FULL_SWEEP:
        return {
            "us_files": ["US_complete_cal.nrrd"],
            "perturb": False,
            "n_runs": 30
        }
    
    if exp_type == ExperimentType.MISSING_DATA:
        return {
            "us_files": ["US_full_L3_dropoutref_cal.nrrd"],
            "perturb": False,
            "n_runs": 10
        }
    
    if exp_type == ExperimentType.ROBUSTNESS:
        return {
            "us_files": [
                "US_complete_cal.nrrd",
                "US_full_L3_dropoutref_cal.nrrd"
            ],
            "perturb": True,
            "n_runs": 30
        }


def init_results():
    """initialize results container for metrics"""
    return {
        "initial_tre": [],
        "final_tre": [],
        "runtime_sec": [],
        "success": [],
        "per_vertebra_success": [],
        "loss_history": [],
        "mean_sim_history": [],
        "axes_penalty_history": [],
        "ivd_loss_history": [],
        "facet_loss_history": []
    }


def success_from_tre(tre_dict, thresh=2.0):
    """check success per vertebra and overall"""
    per_vertebra = {}
    for case, tre in tre_dict.items():
        if tre is None:
            per_vertebra[case] = None
        else:
            per_vertebra[case] = tre < thresh
    
    # overall success = all vertebrae successful
    vals = [v for v in per_vertebra.values() if v is not None]
    overall = all(vals) if vals else False
    
    return overall, per_vertebra


def compute_case_tre(flat_params, K, case_names, case_landmarks, centers):
    tre_results = {}
    for k, case in enumerate(case_names):
        ct_lms, us_lms = case_landmarks[k]
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

        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k:6*(k+1)].tolist())
        tx_inv = tx.GetInverse()
        transforms_list.append(tx_inv)
    
        sampled_positions = sampled_positions_list[k].to(device=device, dtype=torch.float64) 

        M = sitk_euler_to_matrix(tx_inv) 
        M_torch = torch.from_numpy(M).to(device=device, dtype=torch.float64) 
        
        N = sampled_positions.shape[0]
        pts_h = torch.cat([sampled_positions, torch.ones((N, 1), device=device, dtype=sampled_positions.dtype)], dim=1)
        moved_positions = (pts_h @ M_torch.T)[:, :3]

        moving_vals = fixed_parser.sample_at_physical_points_gpu(moved_positions)
        moving_intensities = moving_vals.float()

        sim = torch.mean(moving_intensities)
        total_sim += sim
        
        ct_centroid = torch.tensor(case_centroids[k], device=device, dtype=torch.float32)
        moved_centroid = torch.tensor(tx_inv.TransformPoint(ct_centroid.cpu().numpy().tolist()), device=device)
        moved_centroids.append(moved_centroid.cpu().numpy())

    mean_sim = total_sim / float(K)

    lambda_axes = 0.01 # 0.01
    axes_margins = {
        'LM': 2.0,
        'AP': 2.0,
        'SI': 5.0,
        'LM_rot': np.deg2rad(15),
        'AP_rot': np.deg2rad(6),
        'SI_rot': np.deg2rad(2)
    }

    axes_penalty = compute_inter_vertebral_displacement_penalty(
        moved_centroids, case_centroids, case_axes, transforms_list, axes_margins
    )

    lambda_ivd = 0.001 # 0.001
    ivd_loss, ivd_metrics = compute_ivd_collision_loss(pairings, transforms_list, case_names)
    
    lambda_facet = 0.000
    facet_loss, facet_metrics = compute_facet_collision_loss(facet_pairings, transforms_list, case_names)

    sim_weight = 1.0
    total_loss = (
        sim_weight * -float(mean_sim) + 
        (lambda_axes * axes_penalty) +
        (lambda_ivd * ivd_loss) +
        (lambda_facet * facet_loss)
    )

    return (
        total_loss,
        float(mean_sim * sim_weight),
        float(axes_penalty * lambda_axes),
        float(ivd_loss * lambda_ivd),
        float(facet_loss * lambda_facet),
        ivd_metrics
    )


# SINGLE REGISTRATION WRAPPER
def run_single_registration(fixed_file, cases_dir, mesh_dir, output_dir, case_names, 
    apply_perturbation, rng_seed=None, K=None, pairings=None,facet_pairings=None,track_metrics=False,
    save_transforms=False):

    
    # set random seed if provided
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()
    

    # PERTURBATION
    if apply_perturbation:
        rot = np.deg2rad(rng.uniform(-10.0, 10.0, size=3))
        trans = rng.uniform(-5.0, 5.0, size=3)
        global_perturbation = np.concatenate([rot, trans])
        print(f"\nApplied random perturbation (seed={rng_seed}):")
        print(f"  Rotation (deg): {np.rad2deg(global_perturbation[:3])}")
        print(f"  Translation (mm): {global_perturbation[3:]}")
    else:
        global_perturbation = np.zeros(6)
    

    # LOAD DATA
    fixed_parser = PyNrrdParser(fixed_file)
    
    moving_parsers = []
    sampled_positions_list = []
    centers = []
    case_landmarks = []
    case_axes = []
    case_centroids = []
    
    for case in case_names:
        case_path = os.path.join(cases_dir, case)
        moving_file = os.path.join(case_path, 'moving.nrrd')
        
        moving_parser = PyNrrdParser(moving_file)
        moving_parsers.append(moving_parser)
        moving_tensor = moving_parser.get_tensor(False)
        
        mask = moving_tensor > 0
        ct_mask_indices = torch.stack(torch.where(mask), dim=-1)
        num_vox = ct_mask_indices.shape[0]
        samples_count = min(5000, num_vox)
        
        if samples_count == 0:
            raise RuntimeError(f"No positive voxels found for case {case}")
        
        samples = torch.randint(num_vox, (samples_count,))
        sampled_indices = ct_mask_indices[samples]
        sampled_positions = moving_parser.compute_positions(sampled_indices)
        sampled_positions_list.append(sampled_positions)
        
        moving_img = sitk.ReadImage(moving_file)
        center = np.array(
            moving_img.TransformContinuousIndexToPhysicalPoint(
                np.array(moving_img.GetSize()) / 2.0
            )
        )
        centers.append(center)
        
        # landmarks
        target_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/US_{case}_landmarks.mrk.json"
        source_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/landmarks/CT_{case}_landmarks_intra.mrk.json"
        
        try:
            fixed_lm_parser = SlicerJsonTagParser(target_file)
            moving_lm_parser = SlicerJsonTagParser(source_file)
            ct_lms = moving_lm_parser.extract_landmarks()
            us_lms = fixed_lm_parser.extract_landmarks()
            case_landmarks.append((ct_lms, us_lms))
        except Exception:
            case_landmarks.append((None, None))
        
        # CT axes
        ct_seg_file = os.path.join(cases_dir, case, f'CT_{case}.nrrd')
        LM_axis, AP_axis, SI_axis = compute_ct_axes(ct_seg_file)
        case_axes.append((LM_axis, AP_axis, SI_axis))
        
        # Centroids
        c = compute_centroid(ct_seg_file)
        case_centroids.append(np.array(c))
    
    orig_dists = []
    for k in range(len(case_centroids) - 1):
        d = np.linalg.norm(case_centroids[k] - case_centroids[k + 1])
        orig_dists.append(float(d))
    orig_dists = np.array(orig_dists)
    
    # move to GPU
    sampled_positions_list_gpu = [torch.from_numpy(pos.astype(np.float32)).cuda() 
                                   for pos in sampled_positions_list]
    
    # INITIAL TRE
    tre_before = compute_case_tre(np.tile(global_perturbation, K), K, case_names, case_landmarks, centers)
    
    start_time = time.time()

    # CMA OPTIMIZATION
    x0 = np.tile(global_perturbation, K)
    sigma0 = 0.5
    base_stds = [0.01, 0.01, 0.01, 1.0, 1.0, 1.0]
    cma_stds = base_stds * K
    
    popsize = 100 # was 80
    parents = 20
    lower_per = [-0.4, -0.4, -0.4, -7, -7, -7] # first three are rotation (rad), then translation
    upper_per = [0.4, 0.4, 0.4, 7, 7, 7] # was all 5 and -5
    lower = lower_per * K
    upper = upper_per * K
    
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
        facet_pairings=facet_pairings,
        case_names=case_names,
        device='cuda',
        profile=False
    )
    
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        options={
            'CMA_stds': cma_stds,
            'popsize': popsize,
            'CMA_mu': parents,
            'bounds': [lower, upper],
            'verb_disp': 0,  
            'maxiter': 100, # was 80
            'tolfun': 1e-5,
        }
    )

    if track_metrics:
        loss_history = []
        mean_sim_history = []
        axes_penalty_history = []
        ivd_loss_history = []
        facet_loss_history = []
    
    while not es.stop():
        solutions = es.ask()
        values = []
        
        for sol in solutions:
            val, mean_sim, axes_pen, ivd_loss, facet_loss, _ = partial_eval(sol)
            values.append(val)

            if track_metrics:
                loss_history.append(val)
                mean_sim_history.append(mean_sim)
                axes_penalty_history.append(axes_pen)
                ivd_loss_history.append(ivd_loss)
                facet_loss_history.append(facet_loss)
        
        es.tell(solutions, values)
    

    # FINAL TRE
    best_flat = es.result.xbest
    tre_after = compute_case_tre(best_flat, K, case_names, case_landmarks, centers)

    if save_transforms:
        print("\nSaving transforms for each vertebra...")
        for k, case in enumerate(case_names):
            params = best_flat[6*k:6*(k+1)]
            tx = sitk.Euler3DTransform()
            tx.SetCenter(centers[k].tolist())
            tx.SetParameters(params.tolist())
            
            # Save transform with run_id in filename if doing multiple runs
            if rng_seed is not None and rng_seed > 0:
                out_name = os.path.join(output_dir, f"TransformParameters_groupwise_{case}_run{rng_seed}.h5")
            else:
                out_name = os.path.join(output_dir, f"TransformParameters_groupwise_{case}.h5")
            
            sitk.WriteTransform(tx, out_name)
            print(f"  Wrote transform for {case}: {out_name}")
    
    runtime = time.time() - start_time
    success, per_vertebra_success = success_from_tre(tre_after, SUCCESS_THRESH_MM)

    metrics_dict = None
    if track_metrics:
        metrics_dict = {
            'loss': loss_history,
            'mean_sim': mean_sim_history,
            'axes_penalty': axes_penalty_history,
            'ivd_loss': ivd_loss_history,
            'facet_loss': facet_loss_history
        }
    
    return tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict



# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*NoneType.*check_attribute.*')
    
    # paths
    mesh_dir = '/usr/local/data/elise/pig_data/pig2/Registration/cropped/intra1'
    cases_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/Cases'
    output_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/output_python_cma_group_allcases'
    os.makedirs(output_dir, exist_ok=True)
    
    # get case names
    case_names = sorted([
        name for name in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
    ])
    K = len(case_names)
    
    # precompute IVD and facet pairings (only once)
    print("Computing IVD point pairings...")
    pairings, meshes = compute_adjacent_vertebra_pairings(
        mesh_dir, "_body.vtk", n_sample=30000, n_pairs=100, max_dist=7.0, seed=42
    )
    for mesh in meshes.values():
        mesh.clear_data()
    meshes.clear()
    
    print("Computing facet point pairings...")
    facet_pairings, facet_meshes = compute_adjacent_vertebra_pairings(
        mesh_dir, "_upper.vtk", n_sample=30000, n_pairs=50, max_dist=4.0, seed=42
    )
    for mesh in facet_meshes.values():
        mesh.clear_data()
    facet_meshes.clear()
    
    # convert to numpy
    for k in pairings:
        pairings[k]['L_i'] = np.asarray(pairings[k]['L_i'])
        pairings[k]['L_j'] = np.asarray(pairings[k]['L_j'])
    for k in facet_pairings:
        facet_pairings[k]['L_i'] = np.asarray(facet_pairings[k]['L_i'])
        facet_pairings[k]['L_j'] = np.asarray(facet_pairings[k]['L_j'])
    

    # EXPERIMENT EXECUTION
    settings = get_experiment_settings(EXPERIMENT)
    all_results = defaultdict(init_results)
    
    print(f"\n{'='*70}")
    print(f"Running experiment: {EXPERIMENT.value}")
    print(f"Settings: {settings}")
    print(f"{'='*70}\n")
    
    for us_file in settings["us_files"]:
        print(f"\n{'='*70}")
        print(f"US FILE: {us_file}")
        print(f"{'='*70}")
        
        fixed_file = os.path.join(cases_dir, us_file)
        
        for run_id in range(settings["n_runs"]):
            print(f"\n--- Run {run_id+1}/{settings['n_runs']} ---")

            # track metrics for normal runs
            track_metrics = (EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1)

            
            tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict = run_single_registration(
                fixed_file=fixed_file,
                cases_dir=cases_dir,
                mesh_dir=mesh_dir,
                output_dir=output_dir,
                case_names=case_names,
                apply_perturbation=settings["perturb"],
                rng_seed=run_id,
                K=K,
                pairings=pairings,
                facet_pairings=facet_pairings,
                track_metrics=track_metrics,
                save_transforms=(EXPERIMENT == ExperimentType.NORMAL)
            )
            
            # results
            all_results[us_file]["initial_tre"].append(tre_before)
            all_results[us_file]["final_tre"].append(tre_after)
            all_results[us_file]["runtime_sec"].append(runtime)
            all_results[us_file]["success"].append(success)
            all_results[us_file]["per_vertebra_success"].append(per_vertebra_success)  # Add this line

            
            # summary
            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Success: {success}")
            for case in case_names:
                if tre_before.get(case) is not None:
                    print(f"  {case}: {tre_before[case]:.2f} mm -> {tre_after[case]:.2f} mm")
    

    # PLOTTING CODE (normal runs only)
    if EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1:
        # get metrics from the single run
        for us_file in settings["us_files"]:
            if metrics_dict is not None:
                mean_sim_arr = -1 * np.array(metrics_dict['mean_sim'])
                axes_penalty_arr = np.array(metrics_dict['axes_penalty'])
                ivd_loss_arr = np.array(metrics_dict['ivd_loss'])
                facet_loss_arr = np.array(metrics_dict['facet_loss'])
                loss_arr = np.array(metrics_dict['loss'])
                
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
                
                plot_path = os.path.join("optimization_metrics.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"\n Saved optimization plot to: {plot_path}")




    # SAVE RESULTS
    out_path = os.path.join(output_dir, f"experiment_{EXPERIMENT.value}.json")

    # calculate summary statistics before saving
    summary_stats = {}

    for us_file, results in all_results.items():
        total_runs = len(results['success'])
        
        # success rate
        overall_success_rate = sum(results['success']) / total_runs if total_runs > 0 else 0
        
        # per-vertebra success rates
        vertebra_success_counts = {case: 0 for case in case_names}
        for run_result in results['per_vertebra_success']:
            for case, success_status in run_result.items():
                if success_status is True:
                    vertebra_success_counts[case] += 1
        
        vertebra_success_rates = {
            case: vertebra_success_counts[case] / total_runs if total_runs > 0 else 0
            for case in case_names
        }
        
        # mean final TRE per vertebra
        mean_tre_per_vertebra = {}
        std_tre_per_vertebra = {}
        for case in case_names:
            tre_values = []
            for run_tre in results['final_tre']:
                if case in run_tre and run_tre[case] is not None:
                    tre_values.append(run_tre[case])
            
            if tre_values:
                mean_tre_per_vertebra[case] = float(np.mean(tre_values))
                std_tre_per_vertebra[case] = float(np.std(tre_values))
            else:
                mean_tre_per_vertebra[case] = None
                std_tre_per_vertebra[case] = None
        
        # summary stats
        summary_stats[us_file] = {
            "total_runs": total_runs,
            "overall_success_rate": float(overall_success_rate),
            "mean_runtime_sec": float(np.mean(results['runtime_sec'])),
            "std_runtime_sec": float(np.std(results['runtime_sec'])),
            "vertebra_success_rates": vertebra_success_rates,
            "vertebra_success_counts": vertebra_success_counts,
            "mean_final_tre_per_vertebra": mean_tre_per_vertebra,
            "std_final_tre_per_vertebra": std_tre_per_vertebra
        }

    # add summary to results
    all_results["_summary"] = summary_stats

    # save everything
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f" Saved results to: {out_path}")
    print(f"{'='*70}")

    # print
    for us_file, summary in summary_stats.items():
        print(f"\n{'='*70}")
        print(f"Summary for {us_file}")
        print(f"{'='*70}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Overall success rate: {summary['vertebra_success_counts'][case_names[0]]}/{summary['total_runs']} ({100*summary['overall_success_rate']:.1f}%)")
        print(f"Mean runtime: {summary['mean_runtime_sec']:.1f}s ± {summary['std_runtime_sec']:.1f}s")
        
        print(f"\nPer-vertebra success rates:")
        for case in case_names:
            count = summary['vertebra_success_counts'][case]
            rate = 100 * summary['vertebra_success_rates'][case]
            print(f"  {case}: {count}/{summary['total_runs']} ({rate:.1f}%)")
        
        print(f"\nMean final TRE per vertebra:")
        for case in case_names:
            mean_tre = summary['mean_final_tre_per_vertebra'][case]
            std_tre = summary['std_final_tre_per_vertebra'][case]
            if mean_tre is not None:
                print(f"  {case}: {mean_tre:.2f} ± {std_tre:.2f} mm")
            else:
                print(f"  {case}: No TRE data available")