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
import nrrd

# imports from my files
from utils.file_parser import SlicerJsonTagParser, PyNrrdParser
from utils.helpers import sitk_euler_to_matrix, preprocess_US
from utils.similarity import IntensitySimilarity
from extra.centroid import compute_centroid
from extra.CT_axis import compute_ct_axes
from extra.IVD_points import compute_adjacent_vertebra_pairings

# Gill et al. 2012 implementation functions
from US_simulation import preload_ct_hu_volumes, evaluate_group_gpu_gill


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

class ExperimentType(Enum):
    NORMAL      = "normal"
    FULL_SWEEP  = "full_sweep"
    MISSING_DATA = "missing_data"
    ROBUSTNESS  = "robustness"


# CHANGE THIS TO SELECT EXPERIMENT
EXPERIMENT = ExperimentType.NORMAL
SUCCESS_THRESH_MM = 2.0


def get_experiment_settings(exp_type):
    if exp_type == ExperimentType.NORMAL:
        return {
            "us_files": ["US_full_L3_dropoutref_cal.nrrd"],
            "perturb": True,
            "n_runs": 1
        }
    if exp_type == ExperimentType.FULL_SWEEP:
        return {
            "us_files": ["US_complete_cal.nrrd"],
            "perturb": True,
            "n_runs": 10
        }
    if exp_type == ExperimentType.MISSING_DATA:
        return {
            "us_files": ["US_full_L3_dropoutref_cal.nrrd"],
            "perturb": False,
            "n_runs": 30
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
    return {
        "initial_tre": [],
        "final_tre": [],
        "runtime_sec": [],
        "success": [],
        "per_vertebra_success": [],
        "loss_history": [],
        "mean_sim_history": [],
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
        moved = torch.tensor(
            [tx_inv.TransformPoint(p.tolist()) for p in ct_lms.cpu().numpy()]
        )
        diff = moved - us_lms
        tre_results[case] = float(torch.mean(torch.linalg.norm(diff, dim=1)).item())
    return tre_results


# ============================================================================
# SINGLE REGISTRATION
# ============================================================================

def run_single_registration(
    fixed_file, cases_dir, mesh_dir, output_dir, case_names,
    apply_perturbation, ct_hu_parsers,
    rng_seed=None, K=None,
    track_metrics=False, save_transforms=False
):
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # PERTURBATION
    # ------------------------------------------------------------------
    if apply_perturbation:
        rot   = np.deg2rad(rng.uniform(-10.0, 10.0, size=3))
        trans = rng.uniform(-10.0, 10.0, size=3)
        global_perturbation = np.concatenate([rot, trans])
        print(f"\nApplied random perturbation (seed={rng_seed}):")
        print(f"  Rotation (deg): {np.rad2deg(global_perturbation[:3])}")
        print(f"  Translation (mm): {global_perturbation[3:]}")
    else:
        global_perturbation = np.zeros(6)

    # ------------------------------------------------------------------
    # PRE-PROCESS ULTRASOUND
    # ------------------------------------------------------------------
    print("Preprocessing US image...")
    preprocess_start = time.time()
    enhanced_us_data, us_header = preprocess_US(
        fixed_file, True, method='tophat', sigma=1.0, size=5
    )
    preprocess_time = time.time() - preprocess_start
    print(f"  Preprocessing completed in {preprocess_time:.2f}s")

    temp_us_file = os.path.join(output_dir, 'temp_preprocessed_us.nrrd')
    nrrd.write(temp_us_file, enhanced_us_data, us_header)
    fixed_parser = PyNrrdParser(temp_us_file)

    # ------------------------------------------------------------------
    # LOAD PER-CASE DATA  (landmarks, centers, axes, centroids)
    # ------------------------------------------------------------------
    centers        = []
    case_landmarks = []
    case_axes      = []
    case_centroids = []

    for case in case_names:
        case_path   = os.path.join(cases_dir, case)
        moving_file = os.path.join(case_path, 'moving.nrrd')   # binary mask — used for center/centroid only

        moving_img = sitk.ReadImage(moving_file)
        center = np.array(
            moving_img.TransformContinuousIndexToPhysicalPoint(
                np.array(moving_img.GetSize()) / 2.0
            )
        )
        centers.append(center)

        # Landmarks for TRE
        target_file = (
            f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/"
            f"sofa1/landmarks/US_{case}_landmarks.mrk.json"
        )
        source_file = (
            f"/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/"
            f"sofa1/landmarks/CT_{case}_landmarks_intra.mrk.json"
        )
        try:
            fixed_lm_parser  = SlicerJsonTagParser(target_file)
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

    # ------------------------------------------------------------------
    # INITIAL TRE
    # ------------------------------------------------------------------
    tre_before = compute_case_tre(
        np.tile(global_perturbation, K), K, case_names, case_landmarks, centers
    )

    start_time = time.time()

    # ------------------------------------------------------------------
    # CMA-ES SETUP  (hyperparameters unchanged from original)
    # ------------------------------------------------------------------
    x0        = np.tile(global_perturbation, K)
    sigma0    = 0.25
    base_stds = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
    cma_stds  = base_stds * K

    popsize  = 60
    parents  = 20
    lower_per = [-0.4, -0.4, -0.4, -10, -10, -10]
    upper_per = [ 0.4,  0.4,  0.4,  10,  10,  10]
    lower     = lower_per * K
    upper     = upper_per * K
    max_iter  = 160

    partial_eval = partial(
        evaluate_group_gpu_gill,
        K              = K,
        centers        = centers,
        ct_hu_parsers  = ct_hu_parsers,
        fixed_parser   = fixed_parser,
        case_centroids = case_centroids,
        case_names     = case_names,
        max_iter       = max_iter,
        sigma          = 1.0,      # biomechanical model weight (Gill: best at sigma=1)
        beam_axis      = 1,        # AP axis in DHW layout
        device         = 'cuda',
    )

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        options={
            'CMA_stds': cma_stds,
            'popsize':  popsize,
            'CMA_mu':   parents,
            'bounds':   [lower, upper],
            'verb_disp': 0,
            'maxiter':  max_iter,
            'tolfun':   1e-5,
            'seed':     42
        }
    )

    if track_metrics:
        loss_history     = []
        mean_sim_history = []

    # ------------------------------------------------------------------
    # CMA OPTIMISATION LOOP
    # ------------------------------------------------------------------
    while not es.stop():
        solutions = es.ask()
        values    = []
        iteration = es.countiter

        for sol in solutions:
            val, mean_sim, _, _, _, _ = partial_eval(sol, iteration=iteration)
            values.append(val)

            if track_metrics:
                loss_history.append(val)
                mean_sim_history.append(mean_sim)

        es.tell(solutions, values)

    # ------------------------------------------------------------------
    # FINAL TRE
    # ------------------------------------------------------------------
    best_flat = es.result.xbest
    tre_after = compute_case_tre(best_flat, K, case_names, case_landmarks, centers)

    # ------------------------------------------------------------------
    # PRINT TRE
    # ------------------------------------------------------------------
    print("\nTRE results:")
    for case in case_names:
        before = tre_before.get(case)
        after  = tre_after.get(case)
        if before is not None and after is not None:
            print(f"  {case}: {before:.2f} mm  ->  {after:.2f} mm")
        else:
            print(f"  {case}: no landmarks available")

    # ------------------------------------------------------------------
    # SAVE TRANSFORMS
    # ------------------------------------------------------------------
    if save_transforms:
        print("\nSaving transforms...")
        for k, case in enumerate(case_names):
            params = best_flat[6*k:6*(k+1)]
            tx = sitk.Euler3DTransform()
            tx.SetCenter(centers[k].tolist())
            tx.SetParameters(params.tolist())

            if rng_seed is not None and rng_seed > 0:
                out_name = os.path.join(
                    output_dir, f"GillTransformParameters_groupwise_{case}_run{rng_seed}.h5"
                )
            else:
                out_name = os.path.join(
                    output_dir, f"GillTransformParameters_groupwise_{case}.h5"
                )
            sitk.WriteTransform(tx, out_name)
            print(f"  Wrote: {out_name}")

    runtime = time.time() - start_time
    success, per_vertebra_success = success_from_tre(tre_after, SUCCESS_THRESH_MM)

    metrics_dict = None
    if track_metrics:
        metrics_dict = {
            'loss':     loss_history,
            'mean_sim': mean_sim_history,
        }

    # Clean up temp US file
    if os.path.exists(temp_us_file):
        os.remove(temp_us_file)

    return tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*NoneType.*check_attribute.*')

    # Paths
    mesh_dir   = '/usr/local/data/elise/pig_data/pig2/Registration/cropped/sofa1'
    cases_dir  = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/sofa1/Cases'
    output_dir = '/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/Gill/sofa1/output_python_cma_group_allcases'
    os.makedirs(output_dir, exist_ok=True)

    # Case names
    case_names = sorted([
        name for name in os.listdir(cases_dir)
        if os.path.isdir(os.path.join(cases_dir, name)) and name.startswith('L')
    ])
    K = len(case_names)
    print(f"Found {K} cases: {case_names}")

    # ------------------------------------------------------------------
    # Load CT HU volumes once (shared across all runs)
    # ------------------------------------------------------------------
    print("\nLoading CT HU volumes...")
    ct_hu_parsers = preload_ct_hu_volumes(cases_dir, case_names, device='cuda')

    # ------------------------------------------------------------------
    # EXPERIMENT
    # ------------------------------------------------------------------
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

            track_metrics = (
                EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1
            )

            tre_before, tre_after, runtime, success, per_vertebra_success, metrics_dict = \
                run_single_registration(
                    fixed_file         = fixed_file,
                    cases_dir          = cases_dir,
                    mesh_dir           = mesh_dir,
                    output_dir         = output_dir,
                    case_names         = case_names,
                    apply_perturbation = settings["perturb"],
                    ct_hu_parsers      = ct_hu_parsers,
                    rng_seed           = None,
                    K                  = K,
                    track_metrics      = track_metrics,
                    save_transforms    = (EXPERIMENT == ExperimentType.NORMAL),
                )

            all_results[us_file]["initial_tre"].append(tre_before)
            all_results[us_file]["final_tre"].append(tre_after)
            all_results[us_file]["runtime_sec"].append(runtime)
            all_results[us_file]["success"].append(success)
            all_results[us_file]["per_vertebra_success"].append(per_vertebra_success)

            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Success: {success}")

    # ------------------------------------------------------------------
    # PLOT  (normal single run only)
    # ------------------------------------------------------------------
    if EXPERIMENT == ExperimentType.NORMAL and settings["n_runs"] == 1:
        for us_file in settings["us_files"]:
            if metrics_dict is not None:
                loss_arr     = np.array(metrics_dict['loss'])
                mean_sim_arr = np.array(metrics_dict['mean_sim'])

                plt.figure(figsize=(10, 6))
                plt.plot(loss_arr,     '.', label='Total Loss (BCLC2)',  linestyle='None')
                plt.plot(mean_sim_arr, 'o', label='LC2 Similarity',      linestyle='None')
                plt.xlabel('CMA Evaluation Step')
                plt.ylabel('Value')
                plt.title('CMA Optimisation Metrics (Gill et al. BCLC2)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plot_path = os.path.join(output_dir, "optimization_metrics.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"\nSaved plot to: {plot_path}")

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    out_path     = os.path.join(output_dir, f"experiment_{EXPERIMENT.value}.json")
    summary_stats = {}

    for us_file, results in all_results.items():
        total_runs = len(results['success'])

        overall_success_rate = (
            sum(results['success']) / total_runs if total_runs > 0 else 0
        )

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

        summary_stats[us_file] = {
            "total_runs":                  total_runs,
            "overall_success_rate":        float(overall_success_rate),
            "mean_runtime_sec":            float(np.mean(results['runtime_sec'])),
            "std_runtime_sec":             float(np.std(results['runtime_sec'])),
            "vertebra_success_rates":      vertebra_success_rates,
            "vertebra_success_counts":     vertebra_success_counts,
            "mean_final_tre_per_vertebra": mean_tre_per_vertebra,
            "std_final_tre_per_vertebra":  std_tre_per_vertebra,
        }

    all_results["_summary"] = summary_stats

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved results to: {out_path}")
    print(f"{'='*70}")

    for us_file, summary in summary_stats.items():
        print(f"\n{'='*70}")
        print(f"Summary for {us_file}")
        print(f"{'='*70}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Overall success rate: "
              f"{summary['vertebra_success_counts'][case_names[0]]}/{summary['total_runs']} "
              f"({100*summary['overall_success_rate']:.1f}%)")
        print(f"Mean runtime: {summary['mean_runtime_sec']:.1f}s "
              f"± {summary['std_runtime_sec']:.1f}s")

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
                print(f"  {case}: no TRE data available")