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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Optuna imports
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# imports from my files
from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix, step_lambda, linear_lambda, sigmoid_ramp, gaussian_lambda, compute_inter_vertebral_displacement_penalty, compute_ivd_collision_loss, compute_facet_collision_loss, preprocess_US
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid
from extra.CT_axis import compute_ct_axes
from extra.IVD_points import compute_adjacent_vertebra_pairings
import nrrd


# ─── multiprocessing worker 
_partial_eval_global = None

def _init_worker(eval_fn):
    global _partial_eval_global
    _partial_eval_global = eval_fn

def _call_eval(args):
    sol, iteration = args
    return _partial_eval_global(sol, iteration=iteration)
# ─────────────────────────────────────────────────────────────────────────────


# EXPERIMENT CONFIGURATION
class ExperimentType(Enum):
    OPTUNA_TUNING = "optuna_tuning"
    VALIDATE_BEST = "validate_best"


# CHANGE THIS TO SELECT EXPERIMENT
EXPERIMENT = ExperimentType.OPTUNA_TUNING
SUCCESS_THRESH_MM = 2.0


def get_experiment_settings(exp_type):
    if exp_type == ExperimentType.OPTUNA_TUNING:
        return {
            "us_files": ["US_missing_combined.nrrd"],
            "perturb": True,
            "n_trials_per_optuna": 5,
            "n_optuna_trials": 50,
        }
    
    if exp_type == ExperimentType.VALIDATE_BEST:
        return {
            "us_files": ["US_complete_cal.nrrd"],
            "perturb": True,
            "n_runs": 20,
        }


def init_results():
    return {
        "initial_tre": [],
        "final_tre": [],
        "runtime_sec": [],
        "success": [],
        "per_vertebra_success": [],
    }


def success_from_tre(tre_dict, thresh=2.0):
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
                       lambda_axes_final, lambda_ivd_final, lambda_facet,
                       axes_start_frac, ivd_start_frac,
                       axes_margins,
                       device='cpu', profile=False):
    """CPU-only version used by worker processes."""

    total_sim = 0.0
    moved_centroids = []
    transforms_list = []

    for k in range(K):
        tx = sitk.Euler3DTransform()
        tx.SetCenter(centers[k].tolist())
        tx.SetParameters(flat_params[6*k:6*(k+1)].tolist())
        tx_inv = tx.GetInverse()
        transforms_list.append(tx_inv)

        sampled_positions = sampled_positions_list[k]  # numpy (N,3)

        M = sitk_euler_to_matrix(tx_inv)
        N = sampled_positions.shape[0]
        pts_h = np.concatenate([sampled_positions, np.ones((N, 1))], axis=1)
        moved_positions = (pts_h @ M.T)[:, :3]

        # CPU trilinear interpolation
        moving_intensities = fixed_parser.sample_at_physical_points(moved_positions)

        valid_mask = moving_intensities > 0
        n_valid = valid_mask.sum()
        if n_valid > 0:
            total_sim += float(np.mean(moving_intensities[valid_mask]))

        # move centroids
        ct_centroid = np.array(case_centroids[k], dtype=np.float64)
        moved_centroid = np.array(tx_inv.TransformPoint(ct_centroid.tolist()))
        moved_centroids.append(moved_centroid)

    mean_sim = total_sim / float(K)

    lambda_axes = linear_lambda(iteration, max_iter, lambda_axes_final, axes_start_frac)
    axes_penalty = compute_inter_vertebral_displacement_penalty(
        moved_centroids, case_centroids, case_axes, transforms_list, axes_margins
    )

    lambda_ivd_val = linear_lambda(iteration, max_iter, lambda_ivd_final, ivd_start_frac)
    ivd_loss, ivd_metrics = compute_ivd_collision_loss(pairings, transforms_list, case_names)

    facet_loss, _ = compute_facet_collision_loss(facet_pairings, transforms_list, case_names)

    total_loss = (
        -float(mean_sim) +
        (lambda_axes    * axes_penalty) +
        (lambda_ivd_val * ivd_loss) +
        (lambda_facet   * facet_loss)
    )

    return (
        total_loss,
        float(mean_sim),
        float(axes_penalty * lambda_axes),
        float(ivd_loss     * lambda_ivd_val),
        float(facet_loss   * lambda_facet),
        ivd_metrics
    )


def run_single_registration(fixed_file, cases_dir, mesh_dir, output_dir, case_names,
    apply_perturbation, rng_seed=None, K=None, pairings=None, facet_pairings=None,
    save_transforms=False, hyperparams=None):

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # Extract hyperparameters or use defaults
    if hyperparams is None:
        sigma0 = 0.5
        base_stds = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
        popsize = 80
        max_iter = 120
        lambda_axes_final = 0.01
        lambda_ivd_final = 0.002
        lambda_facet = 0.0
        axes_start_frac = 0.25
        ivd_start_frac = 0.25
        axes_margins = {
            'LM': 2.0, 'AP': 2.0, 'SI': 2.0,
            'LM_rot': np.deg2rad(10),
            'AP_rot': np.deg2rad(6),
            'SI_rot': np.deg2rad(2)
        }
    else:
        sigma0 = hyperparams['sigma0']
        base_stds = [
            hyperparams['base_std_rot'],
            hyperparams['base_std_rot'],
            hyperparams['base_std_rot'],
            hyperparams['base_std_trans'],
            hyperparams['base_std_trans'],
            hyperparams['base_std_trans']
        ]
        popsize = hyperparams.get('popsize', 80)
        max_iter = hyperparams.get('max_iter', 120)
        lambda_axes_final = hyperparams['lambda_axes_final']
        lambda_ivd_final  = hyperparams['lambda_ivd_final']
        lambda_facet      = hyperparams['lambda_facet']
        axes_start_frac   = hyperparams['axes_start_frac']
        ivd_start_frac    = hyperparams['ivd_start_frac']
        axes_margins = {
            'LM':     hyperparams['margin_LM'],
            'AP':     hyperparams['margin_AP'],
            'SI':     hyperparams['margin_SI'],
            'LM_rot': np.deg2rad(hyperparams['margin_LM_rot_deg']),
            'AP_rot': np.deg2rad(hyperparams['margin_AP_rot_deg']),
            'SI_rot': np.deg2rad(hyperparams['margin_SI_rot_deg'])
        }

    # PERTURBATION
    if apply_perturbation:
        rot   = np.deg2rad(rng.uniform(-10.0, 10.0, size=3))
        trans = rng.uniform(-10.0, 10.0, size=3)
        global_perturbation = np.concatenate([rot, trans])
    else:
        global_perturbation = np.zeros(6)

    # PRE-PROCESS ULTRASOUND IMAGE
    print("Preprocessing US image...")
    preprocess_start = time.time()
    enhanced_us_data, us_header = preprocess_US(fixed_file, False, method='tophat', sigma=1.0, size=5)
    preprocess_time = time.time() - preprocess_start
    print(f"  Preprocessing completed in {preprocess_time:.2f}s")

    temp_us_file = os.path.join(output_dir, f'temp_preprocessed_us_{rng_seed}.nrrd')
    nrrd.write(temp_us_file, enhanced_us_data, us_header)
    fixed_parser = PyNrrdParser(temp_us_file)

    moving_parsers = []
    sampled_positions_list = []   # numpy (N,3) arrays — no CUDA, picklable
    centers = []
    case_landmarks = []
    case_axes = []
    case_centroids = []

    for case in case_names:
        case_path   = os.path.join(cases_dir, case)
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
        sampled_indices   = ct_mask_indices[samples]
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
    x0       = np.tile(global_perturbation, K)
    cma_stds = base_stds * K
    parents  = popsize // 4

    lower_per = [-0.4, -0.4, -0.4, -10, -10, -10]
    upper_per = [ 0.4,  0.4,  0.4,  10,  10,  10]
    lower = lower_per * K
    upper = upper_per * K

    partial_eval = partial(
        evaluate_group_cpu,              # CPU version — picklable
        K=K,
        centers=centers,
        sampled_positions_list=sampled_positions_list,  # numpy, no CUDA
        moving_parsers=moving_parsers,
        fixed_parser=fixed_parser,
        case_centroids=case_centroids,
        orig_dists=orig_dists,
        case_axes=case_axes,
        pairings=pairings,
        facet_pairings=facet_pairings,
        case_names=case_names,
        max_iter=max_iter,
        lambda_axes_final=lambda_axes_final,
        lambda_ivd_final=lambda_ivd_final,
        lambda_facet=lambda_facet,
        axes_start_frac=axes_start_frac,
        ivd_start_frac=ivd_start_frac,
        axes_margins=axes_margins,
        device='cpu',
        profile=False
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
            'seed':      42
        }
    )

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

            if rng_seed is not None and rng_seed > 0:
                out_name = os.path.join(output_dir, f"TransformParameters_groupwise_{case}_run{rng_seed}.h5")
            else:
                out_name = os.path.join(output_dir, f"TransformParameters_groupwise_{case}.h5")

            sitk.WriteTransform(tx, out_name)
            print(f"  Wrote transform for {case}: {out_name}")

    runtime = time.time() - start_time
    success, per_vertebra_success = success_from_tre(tre_after, SUCCESS_THRESH_MM)

    if os.path.exists(temp_us_file):
        os.remove(temp_us_file)

    return tre_before, tre_after, runtime, success, per_vertebra_success


# ============================================================================
# OPTUNA INTEGRATION  (unchanged from original)
# ============================================================================

def create_objective_function(fixed_file, cases_dir, mesh_dir, output_dir,
                              case_names, K, pairings, facet_pairings, n_trials_per_optuna):

    def objective(trial):

        hyperparams = {
            'sigma0':             trial.suggest_float('sigma0', 0.1, 2.0, log=True),
            'base_std_rot':       trial.suggest_float('base_std_rot', 0.005, 0.05, log=True),
            'base_std_trans':     trial.suggest_float('base_std_trans', 0.1, 2.0, log=True),
            'popsize':            trial.suggest_int('popsize', 40, 120, step=20),
            'max_iter':           trial.suggest_int('max_iter', 80, 200, step=20),
            'lambda_axes_final':  trial.suggest_float('lambda_axes_final', 0.001, 0.05, log=True),
            'lambda_ivd_final':   trial.suggest_float('lambda_ivd_final', 0.0001, 0.01, log=True),
            'lambda_facet':       trial.suggest_float('lambda_facet', 0.0, 0.005),
            'axes_start_frac':    trial.suggest_float('axes_start_frac', 0.0, 0.5),
            'ivd_start_frac':     trial.suggest_float('ivd_start_frac', 0.0, 0.5),
            'margin_LM':          trial.suggest_float('margin_LM', 1.0, 4.0),
            'margin_AP':          trial.suggest_float('margin_AP', 1.0, 4.0),
            'margin_SI':          trial.suggest_float('margin_SI', 1.0, 4.0),
            'margin_LM_rot_deg':  trial.suggest_float('margin_LM_rot_deg', 5, 15),
            'margin_AP_rot_deg':  trial.suggest_float('margin_AP_rot_deg', 3, 10),
            'margin_SI_rot_deg':  trial.suggest_float('margin_SI_rot_deg', 1, 5),
        }

        successes = []
        final_tres = []
        runtimes = []
        per_vertebra_successes = {case: [] for case in case_names}

        for trial_idx in range(n_trials_per_optuna):
            try:
                tre_before, tre_after, runtime, success, per_vertebra_success = run_single_registration(
                    fixed_file=fixed_file,
                    cases_dir=cases_dir,
                    mesh_dir=mesh_dir,
                    output_dir=output_dir,
                    case_names=case_names,
                    apply_perturbation=True,
                    rng_seed=trial.number * 1000 + trial_idx,
                    K=K,
                    pairings=pairings,
                    facet_pairings=facet_pairings,
                    save_transforms=False,
                    hyperparams=hyperparams
                )

                successes.append(int(success))
                runtimes.append(runtime)

                valid_tres = [tre for tre in tre_after.values() if tre is not None]
                if valid_tres:
                    final_tres.extend(valid_tres)

                for case, case_success in per_vertebra_success.items():
                    if case_success is not None:
                        per_vertebra_successes[case].append(int(case_success))

                if trial_idx > 0:
                    intermediate_success_rate = sum(successes) / len(successes)
                    trial.report(intermediate_success_rate, trial_idx)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"Trial {trial.number}, run {trial_idx} failed: {e}")
                continue

        if not successes:
            return 0.0

        success_rate = sum(successes) / len(successes)
        mean_tre     = np.mean(final_tres) if final_tres else 100.0
        mean_runtime = np.mean(runtimes)

        vertebra_success_rates = {
            case: np.mean(successes_list) if successes_list else 0.0
            for case, successes_list in per_vertebra_successes.items()
        }
        min_vertebra_success = min(vertebra_success_rates.values()) if vertebra_success_rates else 0.0

        trial.set_user_attr('mean_tre',              float(mean_tre))
        trial.set_user_attr('mean_runtime',          float(mean_runtime))
        trial.set_user_attr('min_vertebra_success',  float(min_vertebra_success))
        for case, rate in vertebra_success_rates.items():
            trial.set_user_attr(f'success_rate_{case}', float(rate))

        objective_value = (
            -1.0 * success_rate +
             0.01 * mean_tre +
             0.001 * mean_runtime +
             0.2 * (1 - min_vertebra_success)
        )

        return objective_value

    return objective


def save_best_params(study, study_dir):
    best_params_file = study_dir / "best_params.json"
    result = {
        "best_trial":      study.best_trial.number,
        "best_value":      float(study.best_value),
        "best_params":     study.best_params,
        "best_user_attrs": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in study.best_trial.user_attrs.items()},
        "n_trials":        len(study.trials)
    }
    with open(best_params_file, 'w') as f:
        json.dump(result, f, indent=2)


def save_study_results(study, output_file):
    results = {
        "study_name":      study.study_name,
        "n_trials":        len(study.trials),
        "best_trial":      study.best_trial.number,
        "best_value":      float(study.best_value),
        "best_params":     study.best_params,
        "best_user_attrs": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in study.best_trial.user_attrs.items()},
        "all_trials":      []
    }

    for trial in study.trials:
        trial_data = {
            "number":     trial.number,
            "value":      float(trial.value) if trial.value is not None else None,
            "params":     trial.params,
            "user_attrs": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in trial.user_attrs.items()},
            "state":      trial.state.name,
            "duration":   trial.duration.total_seconds() if trial.duration else None
        }
        results["all_trials"].append(trial_data)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed results to: {output_file}")


def generate_optuna_visualizations(study, study_dir, study_name):
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )

        viz_dir = study_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        fig = plot_optimization_history(study)
        fig.write_html(viz_dir / f"{study_name}_history.html")

        fig = plot_param_importances(study)
        fig.write_html(viz_dir / f"{study_name}_importance.html")

        fig = plot_parallel_coordinate(study)
        fig.write_html(viz_dir / f"{study_name}_parallel.html")

        fig = plot_slice(study)
        fig.write_html(viz_dir / f"{study_name}_slice.html")

        print(f"\nSaved visualizations to: {viz_dir}")
    except ImportError:
        print("\nWarning: plotly not installed, skipping visualizations")
        print("Install with: pip install plotly kaleido")


def run_hyperparameter_tuning(fixed_file, cases_dir, mesh_dir, output_dir,
                               case_names, K, pairings, facet_pairings,
                               n_trials, n_trials_per_optuna, study_name="vertebra_registration"):

    study_dir    = Path(output_dir) / "optuna_studies"
    study_dir.mkdir(exist_ok=True)
    storage_name = f"sqlite:///{study_dir / study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    objective = create_objective_function(
        fixed_file, cases_dir, mesh_dir, output_dir,
        case_names, K, pairings, facet_pairings, n_trials_per_optuna
    )

    print(f"\n{'='*70}")
    print(f"Starting Optuna hyperparameter optimization")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage_name}")
    print(f"Target trials: {n_trials}")
    print(f"Runs per trial: {n_trials_per_optuna}")
    print(f"{'='*70}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: save_best_params(study, study_dir)
        ]
    )

    print(f"\n{'='*70}")
    print("Optimization completed!")
    print(f"{'='*70}")
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    results_file = study_dir / f"{study_name}_results.json"
    save_study_results(study, results_file)
    generate_optuna_visualizations(study, study_dir, study_name)

    return study


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

    settings   = get_experiment_settings(EXPERIMENT)
    fixed_file = os.path.join(cases_dir, settings["us_files"][0])

    # ========================================================================
    # EXPERIMENT: OPTUNA TUNING
    # ========================================================================
    if EXPERIMENT == ExperimentType.OPTUNA_TUNING:
        study = run_hyperparameter_tuning(
            fixed_file=fixed_file,
            cases_dir=cases_dir,
            mesh_dir=mesh_dir,
            output_dir=output_dir,
            case_names=case_names,
            K=K,
            pairings=pairings,
            facet_pairings=facet_pairings,
            n_trials=settings["n_optuna_trials"],
            n_trials_per_optuna=settings["n_trials_per_optuna"],
            study_name="vertebra_registration_v1"
        )

    # ========================================================================
    # EXPERIMENT: VALIDATE BEST HYPERPARAMETERS
    # ========================================================================
    elif EXPERIMENT == ExperimentType.VALIDATE_BEST:
        study_dir       = Path(output_dir) / "optuna_studies"
        best_params_file = study_dir / "best_params.json"

        if not best_params_file.exists():
            raise FileNotFoundError(f"Best parameters file not found: {best_params_file}")

        with open(best_params_file, 'r') as f:
            best_data = json.load(f)

        best_hyperparams = best_data['best_params']

        print(f"\n{'='*70}")
        print("Running validation with best parameters")
        print(f"{'='*70}")
        print("\nBest hyperparameters:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")

        validation_results = init_results()

        for run_id in range(settings["n_runs"]):
            print(f"\n--- Validation Run {run_id+1}/{settings['n_runs']} ---")

            tre_before, tre_after, runtime, success, per_vertebra_success = run_single_registration(
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
                hyperparams=best_hyperparams,
                save_transforms=(run_id == 0)
            )

            validation_results["initial_tre"].append(tre_before)
            validation_results["final_tre"].append(tre_after)
            validation_results["runtime_sec"].append(runtime)
            validation_results["success"].append(success)
            validation_results["per_vertebra_success"].append(per_vertebra_success)

            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Success: {success}")
            for case in case_names:
                if tre_before.get(case) is not None:
                    print(f"  {case}: {tre_before[case]:.2f} mm -> {tre_after[case]:.2f} mm")

        total_runs           = len(validation_results['success'])
        overall_success_rate = sum(validation_results['success']) / total_runs

        vertebra_success_counts = {case: 0 for case in case_names}
        for run_result in validation_results['per_vertebra_success']:
            for case, success_status in run_result.items():
                if success_status is True:
                    vertebra_success_counts[case] += 1

        vertebra_success_rates = {
            case: vertebra_success_counts[case] / total_runs
            for case in case_names
        }

        mean_tre_per_vertebra = {}
        std_tre_per_vertebra  = {}
        for case in case_names:
            tre_values = [
                run_tre[case]
                for run_tre in validation_results['final_tre']
                if case in run_tre and run_tre[case] is not None
            ]
            if tre_values:
                mean_tre_per_vertebra[case] = float(np.mean(tre_values))
                std_tre_per_vertebra[case]  = float(np.std(tre_values))
            else:
                mean_tre_per_vertebra[case] = None
                std_tre_per_vertebra[case]  = None

        summary_stats = {
            "best_hyperparams":            best_hyperparams,
            "total_runs":                  total_runs,
            "overall_success_rate":        float(overall_success_rate),
            "mean_runtime_sec":            float(np.mean(validation_results['runtime_sec'])),
            "std_runtime_sec":             float(np.std(validation_results['runtime_sec'])),
            "vertebra_success_rates":      vertebra_success_rates,
            "vertebra_success_counts":     vertebra_success_counts,
            "mean_final_tre_per_vertebra": mean_tre_per_vertebra,
            "std_final_tre_per_vertebra":  std_tre_per_vertebra
        }

        val_path = os.path.join(output_dir, "optuna_validation_results.json")
        with open(val_path, 'w') as f:
            json.dump({"summary": summary_stats, "detailed_results": validation_results}, f, indent=2)

        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total runs: {total_runs}")
        print(f"Overall success rate: {int(overall_success_rate * total_runs)}/{total_runs} ({100*overall_success_rate:.1f}%)")
        print(f"Mean runtime: {summary_stats['mean_runtime_sec']:.1f}s ± {summary_stats['std_runtime_sec']:.1f}s")

        print(f"\nPer-vertebra success rates:")
        for case in case_names:
            count = vertebra_success_counts[case]
            rate  = 100 * vertebra_success_rates[case]
            print(f"  {case}: {count}/{total_runs} ({rate:.1f}%)")

        print(f"\nMean final TRE per vertebra:")
        for case in case_names:
            mean_tre = mean_tre_per_vertebra[case]
            std_tre  = std_tre_per_vertebra[case]
            if mean_tre is not None:
                print(f"  {case}: {mean_tre:.2f} ± {std_tre:.2f} mm")
            else:
                print(f"  {case}: No TRE data available")

        print(f"\n{'='*70}")
        print(f"Validation results saved to: {val_path}")
        print(f"{'='*70}")