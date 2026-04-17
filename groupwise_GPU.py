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
from concurrent.futures import ProcessPoolExecutor
# imports from my files
from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix,step_lambda,linear_lambda, sigmoid_ramp, gaussian_lambda, compute_inter_vertebral_displacement_penalty, compute_ivd_collision_loss, compute_facet_collision_loss, preprocess_US, compute_case_angular_error
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid
from extra.CT_axis import compute_ct_axes
from extra.IVD_points import compute_adjacent_vertebra_pairings
import nrrd


# multiprocessing worker
_partial_eval_global = None

def _init_worker(eval_fn):
    global _partial_eval_global
    _partial_eval_global = eval_fn

def _call_eval(args):
    sol, iteration = args
    return _partial_eval_global(sol, iteration=iteration)


# EXPERIMENT CONFIGURATION
class ExperimentType(Enum):
    NORMAL = "normal"
    FULL_SWEEP = "full_sweep"
    MISSING_DATA = "missing_data"
    SLICE_DATA = "slice_data"
    ROBUSTNESS = "robustness"


# CHANGE THIS TO SELECT EXPERIMENT
EXPERIMENT = ExperimentType.MISSING_DATA
SUCCESS_THRESH_MM = 2.01


def get_experiment_settings(exp_type):
    """define experiment-specific settings"""
    if exp_type == ExperimentType.FULL_SWEEP:
        return {
            "us_files": ["US_complete_cal.nrrd"],
            "perturb": True,
            "n_runs": 30
        }
    
    if exp_type == ExperimentType.MISSING_DATA:
        return {
            "us_files": ["US_full_L3_dropoutref_cal.nrrd"], 
            "perturb": True,
            "n_runs": 30
        }
    
    if exp_type == ExperimentType.SLICE_DATA:
        return {
            "us_files": ["US_missing_combined.nrrd"],
            "perturb": True,
            "n_runs": 30
        }
    
    if exp_type == ExperimentType.ROBUSTNESS:
        return {
            "us_files": [
                "US_complete_cal.nrrd",
                "US_full_L3_dropoutref_cal.nrrd",
                "US_missing_combined.nrrd"
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
        "angular_errors": [],
        "loss_history": [],
        "mean_sim_history": [],
        "axes_penalty_history": [],
        "ivd_loss_history": [],
        "facet_loss_history": []
    }


def success_from_tre(tre_dict, thresh=2.01):
    """check success per vertebra and overall"""
    per_vertebra = {}
    for case, tre in tre_dict.items():
        if tre is None:
            per_vertebra[case] = None
        else:
            per_vertebra[case] = tre < thresh
    
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


def evaluate_group_cpu(flat_params, K, centers, sampled_positions_list,
                       moving_parsers, fixed_parser,
                       case_centroids, orig_dists, case_axes, pairings, facet_pairings,
                       case_names, iteration, max_iter,
                       device='cpu', profile=False, compression_cap=0.7):
    """CPU-only version used by worker processes."""

    total_sim = 0.0
    num_valid_cases = 0
    moved_centroids = []
    transforms_list = []

    for k in range(K):
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k:6*(k+1)].tolist())
        tx_inv = tx.GetInverse()
        transforms_list.append(tx_inv)

        sampled_positions = sampled_positions_list[k]  # already numpy (N,3)

        M = sitk_euler_to_matrix(tx_inv)

        # transform points: (N,4) @ (4,4).T
        N = sampled_positions.shape[0]
        pts_h = np.concatenate([sampled_positions, np.ones((N, 1))], axis=1)  # (N,4)
        moved_positions = (pts_h @ M.T)[:, :3]  # (N,3)

        # CPU trilinear interpolation
        moving_intensities = fixed_parser.sample_at_physical_points(moved_positions)

        valid_mask = moving_intensities > 0
        n_valid = valid_mask.sum()

        if n_valid > 0:
            # Only compute similarity on valid US data points
            sim = float(np.mean(moving_intensities[valid_mask]))
            total_sim += sim
            num_valid_cases += 1

        # sim = float(np.mean(moving_intensities)) # original mean
        # total_sim += sim
        
        # move centroids
        ct_centroid = torch.tensor(case_centroids[k], device=device, dtype=torch.float32)
        moved_centroid = torch.tensor(tx_inv.TransformPoint(ct_centroid.cpu().numpy().tolist()), device=device)
        moved_centroids.append(moved_centroid.cpu().numpy())


    mean_sim = (total_sim / float(K)) # original computation
    # mean_sim = (total_sim / float(num_valid_cases)) ** 1 # other computation , was ** 1.5


    # CONSTRAINT VALUES
    lambda_axes = 0
    # lambda_axes = 0.01
    # lambda_axes = linear_lambda(iteration, max_iter, lambda_final=0.01,  start_frac=0.25)
    # lambda_axes  = linear_lambda(iteration, max_iter, lambda_final=0.01,  start_frac=0.2)


    lambda_ivd = 0
    # lambda_ivd = 0.001
    # lambda_ivd  = linear_lambda(iteration, max_iter, lambda_final=0.002, start_frac=0.25)
    # lambda_ivd   = linear_lambda(iteration, max_iter, lambda_final=0.002, start_frac=0.2)
    
    lambda_facet = 0
    # lambda_facet = 0.001

    axes_margins = {
        'LM': 2.0,
        'AP': 2.0,
        'SI': 2.0,
        'LM_rot': np.deg2rad(10),
        'AP_rot': np.deg2rad(6),
        'SI_rot': np.deg2rad(2)
    }

    axes_penalty = compute_inter_vertebral_displacement_penalty(
        moved_centroids, case_centroids, case_axes, transforms_list, axes_margins
    )

    compression_cap = 0.7
    ivd_loss, ivd_metrics = compute_ivd_collision_loss(
        pairings, transforms_list, case_names, compression_cap=compression_cap
    )
    facet_loss, _ = compute_facet_collision_loss(facet_pairings, transforms_list, case_names)

    total_loss = (
        -float(mean_sim) +
        (lambda_axes  * axes_penalty) +
        (lambda_ivd   * ivd_loss) +
        (lambda_facet * facet_loss)
    )

    return (
        total_loss,
        float(mean_sim),
        float(axes_penalty * lambda_axes),
        float(ivd_loss     * lambda_ivd),
        float(facet_loss   * lambda_facet),
        ivd_metrics
    )


# SINGLE REGISTRATION WRAPPER
def run_single_registration(fixed_file, cases_dir, mesh_dir, output_dir, case_names,
    apply_perturbation, rng_seed=None, K=None, pairings=None, facet_pairings=None,
    track_metrics=False, save_transforms=False):

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # PERTURBATION
    if apply_perturbation:
        rot   = np.deg2rad(rng.uniform(-9.0, 9.0, size=3))
        trans = rng.uniform(-9.0, 9.0, size=3)
        global_perturbation = np.concatenate([rot, trans])
        print(f"\nApplied random perturbation (seed={rng_seed}):")
        print(f"  Rotation (deg): {np.rad2deg(global_perturbation[:3])}")
        print(f"  Translation (mm): {global_perturbation[3:]}")
    else:
        global_perturbation = np.zeros(6)

    # PRE-PROCESS ULTRASOUND IMAGE
    print("Preprocessing US image...")
    preprocess_start = time.time()
    enhanced_us_data, us_header = preprocess_US(fixed_file, method='aniso+tophat', sigma=0.0, size=5)
    preprocess_time = time.time() - preprocess_start
    print(f"  Preprocessing completed in {preprocess_time:.2f}s")

    temp_us_file = os.path.join(output_dir, 'temp_preprocessed_us.nrrd')
    nrrd.write(temp_us_file, enhanced_us_data, us_header)
    fixed_parser = PyNrrdParser(temp_us_file)

    moving_parsers = []
    sampled_positions_list = []   # list of numpy (N,3) arrays — no CUDA, picklable
    centers = []
    case_landmarks = []
    case_axes = []
    case_centroids = []

    for case in case_names:
        case_path  = os.path.join(cases_dir, case)
        moving_file = os.path.join(case_path, 'moving.nrrd')

        moving_parser = PyNrrdParser(moving_file)
        moving_parsers.append(moving_parser)
        moving_tensor = moving_parser.get_tensor(False)

        mask = moving_tensor > 0
        ct_mask_indices = torch.stack(torch.where(mask), dim=-1)
        num_vox = ct_mask_indices.shape[0]
        samples_count = min(6000, num_vox)

        if samples_count == 0:
            raise RuntimeError(f"No positive voxels found for case {case}")

        samples = torch.randint(num_vox, (samples_count,))
        sampled_indices = ct_mask_indices[samples]
        # keep as numpy — workers don't need CUDA
        sampled_positions = moving_parser.compute_positions(sampled_indices)  # numpy (N,3)
        sampled_positions_list.append(sampled_positions)

        moving_img = sitk.ReadImage(moving_file)
        center = np.array(
            moving_img.TransformContinuousIndexToPhysicalPoint(
                np.array(moving_img.GetSize()) / 2.0
            )
        )
        centers.append(center)

        target_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/sofa5/landmarks/US_{case}_landmarks.mrk.json"
        source_file = f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/sofa5/landmarks/CT_{case}_landmarks_intra.mrk.json"

        try:
            fixed_lm_parser  = SlicerJsonTagParser(target_file)
            moving_lm_parser = SlicerJsonTagParser(source_file)
            ct_lms = moving_lm_parser.extract_landmarks()
            us_lms = fixed_lm_parser.extract_landmarks()
            case_landmarks.append((ct_lms, us_lms))
        except Exception:
            case_landmarks.append((None, None))

        ct_seg_file = os.path.join(cases_dir, case, f'CT_{case}.nrrd')
        LM_axis, AP_axis, SI_axis = compute_ct_axes(ct_seg_file)
        case_axes.append((LM_axis, AP_axis, SI_axis))

        c = compute_centroid(ct_seg_file)
        case_centroids.append(np.array(c))

    orig_dists = []
    for k in range(len(case_centroids) - 1):
        d = np.linalg.norm(case_centroids[k] - case_centroids[k + 1])
        orig_dists.append(float(d))
    orig_dists = np.array(orig_dists)

    # INITIAL TRE
    tre_before = compute_case_tre(np.tile(global_perturbation, K), K, case_names, case_landmarks, centers)

    start_time = time.time()

    # CMA SETUP
    x0        = np.tile(global_perturbation, K)
    sigma0    = 0.25
    base_stds = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
    cma_stds  = base_stds * K

    popsize   = 60
    parents   = 20
    lower_per = [-0.4, -0.4, -0.4, -10, -10, -10]
    upper_per = [ 0.4,  0.4,  0.4,  10,  10,  10]
    lower     = lower_per * K
    upper     = upper_per * K
    max_iter  = 160

    partial_eval = partial(
        evaluate_group_cpu,          # CPU version — picklable
        K=K,
        centers=centers,
        sampled_positions_list=sampled_positions_list,   # numpy, no CUDA
        moving_parsers=moving_parsers,
        fixed_parser=fixed_parser,
        case_centroids=case_centroids,
        orig_dists=orig_dists,
        case_axes=case_axes,
        pairings=pairings,
        facet_pairings=facet_pairings,
        case_names=case_names,
        max_iter=max_iter,
        device='cpu',
        profile=False,
        compression_cap = 0.6
    )

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        options={
            'CMA_stds':  cma_stds,
            'popsize':   popsize,
            'CMA_mu':    parents,
            'bounds':    [lower, upper],
            'verb_disp': 0,
            'maxiter':   max_iter,
            'tolfun':    1e-5,
            'seed':      None
        }
    )

    if track_metrics:
        loss_history         = []
        mean_sim_history     = []
        axes_penalty_history = []
        ivd_loss_history     = []
        facet_loss_history   = []

    # PARALLEL CMA LOOP
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=_init_worker,
        initargs=(partial_eval,)
    ) as executor:
        while not es.stop():
            solutions = es.ask()
            iteration = es.countiter

            results = list(executor.map(
                _call_eval,
                [(sol, iteration) for sol in solutions]
            ))
            values = [r[0] for r in results]

            if track_metrics:
                for val, mean_sim, axes_pen, ivd_loss, facet_loss, _ in results:
                    loss_history.append(val)
                    mean_sim_history.append(mean_sim)
                    axes_penalty_history.append(axes_pen)
                    ivd_loss_history.append(ivd_loss)
                    facet_loss_history.append(facet_loss)

            es.tell(solutions, values)

    # FINAL TRE
    best_flat = es.result.xbest
    tre_after = compute_case_tre(best_flat, K, case_names, case_landmarks, centers)

    # angular deviation
    angular_errors = compute_case_angular_error(
        best_flat, K, case_names, case_landmarks, centers
    )
    for case in case_names:
        err = angular_errors.get(case)
        if err is not None:
            print(f"  {case}: angular error = {err:.2f} deg")

    if save_transforms:
        print("\nSaving transforms for each vertebra...")
        for k, case in enumerate(case_names):
            params = best_flat[6*k:6*(k+1)]
            tx = sitk.Euler3DTransform()
            tx.SetCenter(centers[k].tolist())
            tx.SetParameters(params.tolist())

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
            'loss':         loss_history,
            'mean_sim':     mean_sim_history,
            'axes_penalty': axes_penalty_history,
            'facet_loss':   facet_loss_history,
            'ivd_loss':     ivd_loss_history
        }

    if os.path.exists(temp_us_file):
        os.remove(temp_us_file)

    return tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict, angular_errors


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*NoneType.*check_attribute.*')

    mesh_dir   = '/usr/local/data/elise/pig_data/pig2/Registration/cropped/sofa5'
    cases_dir  = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/sofa5/Cases'
    output_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/sofa5/output_python_cma_group_allcases'
    os.makedirs(output_dir, exist_ok=True)

    case_names = sorted([
        name for name in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
    ])
    K = len(case_names)

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

    for k in pairings:
        pairings[k]['L_i'] = np.asarray(pairings[k]['L_i'])
        pairings[k]['L_j'] = np.asarray(pairings[k]['L_j'])
    for k in facet_pairings:
        facet_pairings[k]['L_i'] = np.asarray(facet_pairings[k]['L_i'])
        facet_pairings[k]['L_j'] = np.asarray(facet_pairings[k]['L_j'])

    settings    = get_experiment_settings(EXPERIMENT)
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

            track_metrics = (EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1)

            tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict, angular_errors = run_single_registration(
                fixed_file=fixed_file,
                cases_dir=cases_dir,
                mesh_dir=mesh_dir,
                output_dir=output_dir,
                case_names=case_names,
                apply_perturbation=settings["perturb"],
                rng_seed=None,
                K=K,
                pairings=pairings,
                facet_pairings=facet_pairings,
                track_metrics=track_metrics,
                save_transforms=(EXPERIMENT == ExperimentType.NORMAL)
            )

            all_results[us_file]["initial_tre"].append(tre_before)
            all_results[us_file]["final_tre"].append(tre_after)
            all_results[us_file]["runtime_sec"].append(runtime)
            all_results[us_file]["success"].append(success)
            all_results[us_file]["per_vertebra_success"].append(per_vertebra_success)
            all_results[us_file]["angular_errors"].append(angular_errors) 


            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Success: {success}")
            for case in case_names:
                tre_b = tre_before.get(case)
                tre_a = tre_after.get(case)
                ang   = angular_errors.get(case)
                if tre_b is not None:
                    ang_str = f"  |  angle error: {ang:.2f} deg" if ang is not None else ""
                    print(f"  {case}: {tre_b:.2f} mm -> {tre_a:.2f} mm{ang_str}")

    # PLOTTING (normal single run only)
    if EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1:
        for us_file in settings["us_files"]:
            if metrics_dict is not None:
                mean_sim_arr     = -1 * np.array(metrics_dict['mean_sim'])
                axes_penalty_arr = np.array(metrics_dict['axes_penalty'])
                ivd_loss_arr     = np.array(metrics_dict['ivd_loss'])
                facet_loss_arr   = np.array(metrics_dict['facet_loss'])
                loss_arr         = np.array(metrics_dict['loss'])

                plt.figure(figsize=(10, 6))
                plt.plot(np.where(mean_sim_arr     != 0)[0], mean_sim_arr    [mean_sim_arr     != 0], 'o', label='Mean Similarity',   linestyle='None')
                plt.plot(np.where(axes_penalty_arr != 0)[0], axes_penalty_arr[axes_penalty_arr != 0], 's', label='Axes Penalty',       linestyle='None')
                plt.plot(np.where(ivd_loss_arr     != 0)[0], ivd_loss_arr    [ivd_loss_arr     != 0], '^', label='IVD Spacing Loss',   linestyle='None')
                plt.plot(np.where(loss_arr         != 0)[0], loss_arr        [loss_arr         != 0], '.', label='Total Loss',         linestyle='None')
                plt.plot(np.where(facet_loss_arr   != 0)[0], facet_loss_arr  [facet_loss_arr   != 0], 'v', label='Facet Loss',         linestyle='None')
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
    summary_stats = {}

    for us_file, results in all_results.items():
        total_runs = len(results['success'])
        overall_success_rate = sum(results['success']) / total_runs if total_runs > 0 else 0

        vertebra_success_counts = {case: 0 for case in case_names}
        for run_result in results['per_vertebra_success']:
            for case, success_status in run_result.items():
                if success_status is True:
                    vertebra_success_counts[case] += 1

        vertebra_success_rates = {
            case: vertebra_success_counts[case] / total_runs if total_runs > 0 else 0
            for case in case_names
        }

        mean_tre_per_vertebra = {}
        std_tre_per_vertebra  = {}
        for case in case_names:
            tre_values = [
                run_tre[case]
                for run_tre in results['final_tre']
                if case in run_tre and run_tre[case] is not None
            ]
            if tre_values:
                mean_tre_per_vertebra[case] = float(np.mean(tre_values))
                std_tre_per_vertebra[case]  = float(np.std(tre_values))
            else:
                mean_tre_per_vertebra[case] = None
                std_tre_per_vertebra[case]  = None

        mean_angle_per_vertebra = {}
        std_angle_per_vertebra  = {}
        for case in case_names:
            angle_values = [
                run_ang[case]
                for run_ang in results['angular_errors']
                if case in run_ang and run_ang[case] is not None
            ]
            if angle_values:
                mean_angle_per_vertebra[case] = float(np.mean(angle_values))
                std_angle_per_vertebra[case]  = float(np.std(angle_values))
            else:
                mean_angle_per_vertebra[case] = None
                std_angle_per_vertebra[case]  = None

        summary_stats[us_file] = {
            "total_runs":                  total_runs,
            "overall_success_rate":        float(overall_success_rate),
            "mean_runtime_sec":            float(np.mean(results['runtime_sec'])),
            "std_runtime_sec":             float(np.std(results['runtime_sec'])),
            "vertebra_success_rates":      vertebra_success_rates,
            "vertebra_success_counts":     vertebra_success_counts,
            "mean_final_tre_per_vertebra": mean_tre_per_vertebra,
            "std_final_tre_per_vertebra":  std_tre_per_vertebra,
            "mean_final_angle_per_vertebra": mean_angle_per_vertebra,
            "std_final_angle_per_vertebra":  std_angle_per_vertebra
        }

    all_results["_summary"] = summary_stats

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f" Saved results to: {out_path}")
    print(f"{'='*70}")

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
            rate  = 100 * summary['vertebra_success_rates'][case]
            print(f"  {case}: {count}/{summary['total_runs']} ({rate:.1f}%)")

        print(f"\nMean final TRE per vertebra:")
        for case in case_names:
            mean_tre = summary['mean_final_tre_per_vertebra'][case]
            std_tre  = summary['std_final_tre_per_vertebra'][case]
            if mean_tre is not None:
                print(f"  {case}: {mean_tre:.2f} ± {std_tre:.2f} mm")
            else:
                print(f"  {case}: No TRE data available")

        print(f"\nMean final angular error per vertebra:")
        for case in case_names:
            mean_ang = summary['mean_final_angle_per_vertebra'][case]
            std_ang  = summary['std_final_angle_per_vertebra'][case]
            if mean_ang is not None:
                print(f"  {case}: {mean_ang:.2f} ± {std_ang:.2f} deg")
            else:
                print(f"  {case}: No angular data available")