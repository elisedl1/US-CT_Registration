# Fast Robust Groupwise CT-US Registration for Intraoperative Spinal Interventions

A GPU-accelerated, intensity-based groupwise registration framework for aligning preoperative CT to intraoperative ultrasound (iUS) in spinal surgery. Achieves clinically relevant accuracy (1.65 ± 0.42 mm mean TRE) with clinical speed (~40s) while maintaining robustness under large initial misalignment and missing ultrasound data.

## Overview

This method registers multiple vertebrae simultaneously (L1-L4) while enforcing anatomically realistic spinal configurations through biomechanical constraints.

**Key Innovation**: Unlike traditional pairwise registration, this framework registers all vertebrae simultaneously while preserving inter-vertebral relationships, enabling robust navigation even when individual vertebrae are poorly visible in ultrasound.

## Features

- **Groupwise Multi-Vertebra Registration**: Registers 4 lumbar vertebrae (L1-L4) simultaneously
- **GPU-Accelerated**: Utilizes PyTorch for fast intensity sampling and similarity computation
- **Biomechanical Constraints**: Enforces physically plausible inter-vertebral kinematics
- **Robust to Missing Data**: Maintains accuracy when vertebrae are occluded or missing from US acquisition
- **Clinical Speed**: ~40 seconds per 4-vertebra registration on NVIDIA RTX 3090
- **No Manual Intervention**: Fully automatic—no landmark selection or segmentation required

## Architecture

### Registration Pipeline

```
Preoperative CT               Intraoperative US
      ↓                              ↓
CT Segmentation              US Volume Acquisition
      ↓                              ↓
Surface Extraction    →    GPU Intensity Sampling
      ↓                              ↓
CMA-ES Optimization  ←  Similarity Metric + Constraints
      ↓
Optimized Transforms (T₁, T₂, T₃, T₄)
```

## Installation

### Prerequisites

- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **Hardware**: NVIDIA GPU with ≥8GB VRAM recommended

### Python Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install SimpleITK numpy scipy cma pyvista matplotlib
```

## File Structure

```
project/
├── groupwise_GPU.py           # Main registration script with CMA-ES optimization
├── individual_CPU.py          # Single-vertebra registration (for comparison)
├── utils/
│   ├── file_parser.py        # NRRD/JSON parsing with GPU sampling
│   ├── helpers.py            # Biomechanical constraints & transforms
│   └── similarity.py         # Intensity-based similarity metrics
├── extra/
│   ├── centroid.py           # Vertebral centroid computation
│   ├── CT_axis.py            # Anatomical axis extraction
│   ├── IVD_points.py         # Inter-vertebral disc point pairing
│   └── [various preprocessing scripts]
└── README.md
```

## Usage

### Quick Start

Run groupwise registration on all vertebrae (L1-L4):

```bash
python groupwise_GPU.py
```

The script will:
1. Load CT segmentations and US volumes
2. Sample ~6,000 points per vertebra from CT posterior surfaces
3. Optimize all 4 vertebrae simultaneously using CMA-ES
4. Save transforms as SimpleITK `.h5` files
5. Report initial/final TRE and success rates

### Configuration

Edit the experiment type at the top of `groupwise_GPU.py`:

```python
# Choose experiment type
EXPERIMENT = ExperimentType.NORMAL          # Single run, save transforms
# EXPERIMENT = ExperimentType.FULL_SWEEP    # 30 runs for statistics
# EXPERIMENT = ExperimentType.MISSING_DATA  # Test robustness to occlusions
# EXPERIMENT = ExperimentType.ROBUSTNESS    # Test with perturbations
```

### Data Organization

Organize your data as follows:

```
data/
├── Cases/
│   ├── L1/
│   │   ├── CT_L1.nrrd              # CT segmentation (vertebral body)
│   │   ├── moving.nrrd             # CT posterior surface (for sampling)
│   │   └── fixed.nrrd              # US volume (single vertebra)
│   ├── L2/
│   ├── L3/
│   ├── L4/
│   └── US_complete.nrrd            # Full L1-L4 US sweep
├── landmarks/
│   ├── CT_L1_landmarks.mrk.json    # Slicer fiducial markup
│   ├── US_L1_landmarks.mrk.json
│   └── ...
└── cropped/
    └── intra1/
        ├── L1_body.vtk             # For IVD pairing (optional)
        ├── L1_upper.vtk            # For facet pairing (optional)
        └── ...
```

### Single-Vertebra Registration

For comparison or debugging, run single-vertebra registration:

```bash
python individual_CPU.py
```

This registers one vertebra at a time without biomechanical constraints.

## Method Overview

### Similarity Metric

The registration maximizes normalized ultrasound intensity at transformed CT surface points:

```
L_sim = -1/K ∑(1/N ∑ I_US(T_k^(-1)(p_k,i)))
```

where:
- `K` = number of vertebrae (4)
- `N` = points sampled per vertebra (~6,000)
- `I_US` = normalized ultrasound intensity
- `T_k` = rigid transform for vertebra k
- `p_k,i` = sampled CT surface points

**Why this works**: CT vertebral surfaces correspond to high-intensity regions in ultrasound (bone interfaces).

### Biomechanical Constraints

The framework enforces three types of anatomically realistic constraints:

#### 1. **Inter-Vertebral Disc (IVD) Spacing**

Prevents vertebral collision and enforces realistic disc compression:

- **Directional Consistency**: Penalizes flipping of direction vectors between paired disc surface points
- **Minimum Disc Spacing**: Enforces adaptive lower bound (70% of initial distance)
- **Mean Disc Spacing**: Regularizes global disc height within ±3mm tolerance

#### 2. **Inter-Vertebral Kinematics**

Constrains displacement and rotation between adjacent vertebrae:

**Translational Margins** (measured in inferior vertebra's frame):
- Lateral-Medial: 2.0 mm
- Anterior-Posterior: 2.0 mm
- Superior-Inferior: 5.0 mm

**Rotational Margins**:
- Lateral-Medial axis: 10°
- Anterior-Posterior axis: 6°
- Superior-Inferior axis: 2°

Based on physiological ranges from biomechanics literature.

#### 3. **Facet Joint Constraints**

Maintains realistic facet joint articulation (optional, typically weighted lower than IVD constraints).

### Combined Objective Function

```
L_total = L_sim + λ₁·L_kinematics + λ₂·L_IVD
```

These weights were determined through empirical tuning to balance registration accuracy with anatomical plausibility.

## Optimization

### CMA-ES Configuration

The framework uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for derivative-free optimization.

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Initial σ** | 0.5 | Initial step size |
| **Population Size** | 80 | Candidate solutions per iteration |
| **Parents (μ)** | 20 | Solutions used for recombination |
| **Max Iterations** | 120 | Convergence limit |
| **Parameter Bounds** | [−0.4, 0.4] rad rotation<br>[−8, 8] mm translation | Per-vertebra bounds |
| **Base Standard Deviations** | [0.01, 0.01, 0.01] rad<br>[0.5, 0.5, 0.5] mm | Per-parameter exploration |

**Total Parameters**: 24 (6 per vertebra × 4 vertebrae)


This allows the similarity metric to drive initial alignment before enforcing anatomical realism.

## Validation Results

Results from porcine cadaver experiments with simulated prone-to-supine deformations across four lumbar levels:

### Overall Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean TRE** | 1.65 ± 0.42 mm | Across all conditions (L1-L4) |
| **Runtime** | ~40 seconds | NVIDIA RTX 3090 GPU |
| **Success Rate** | 96.7% | TRE < 2mm (clinical threshold) |
| **Initial TRE** | 5-10 mm | After simulated deformation |

### Per-Scenario Breakdown

| Scenario | Mean TRE (mm) | Success Rate | Description |
|----------|---------------|--------------|-------------|
| **Full US Sweep** | 1.42 ± 0.22 | 100% | Optimal conditions |
| **Missing Slice Sweep** | 1.96 ± 0.40 | 93.3% | Random 2-5 frame dropouts (up to 20%) |
| **Missing Vertebra (L3)** | 1.57 ± 0.64 | 96.7% | Complete absence of one vertebra |

### Robustness to Initialization

With random perturbations (±8° rotation, ±8mm translation):

**With Constraints**:
- Mean TRE: 1.46 ± 1.23 mm
- Success Rate: 93.3%

**Without Constraints**:
- Mean TRE: 7.43 ± 6.29 mm  
- Success Rate: 23.3%

**Conclusion**: Biomechanical constraints are **essential** for reliable intraoperative registration under clinical conditions.

### Comparison with Prior Methods

| Method | Mean TRE (mm) | Runtime | Manual Steps | Handles Missing Data |
|--------|---------------|---------|--------------|----------------------|
| **Ours** | **1.65 ± 0.42** | **~40s** | None | Yes |
| Nagpal et al. [2014] | 1.37 | ~180s | Manual disc points | No |
| Gill et al. [2012] | 0.62-2.26 | ~2580s | Complex biomech model | No |
| Azampour et al. [2024] | 3.67 | ~0.05s | None | No |

## Preprocessing Scripts

The `extra/` directory contains utilities for data preparation:

### Inter-Vertebral Disc (IVD) Point Pairing

```bash
python extra/IVD_points.py
```

Generates paired surface points between adjacent vertebrae for disc spacing constraints. Uses:
- **Uniform sampling**: 30,000 points per mesh
- **Opposing normal filtering**: Keeps only points with normals facing each other
- **Distance threshold**: < 8mm
- **MAD outlier removal**: k=2.5 standard deviations

Output: Dictionary with point pairs `{(L1,L2): {'L_i', 'L_j', 'd0', 'v0'}, ...}`


#### Acoustic Shadow & Reflection Simulation

```bash
python extra/dropout_reflection.py
```

Adds realistic ultrasound artifacts for robustness testing:
- **Shadow**: Exponential attenuation below bright reflectors
- **Reverberation**: Ghost echoes at offset depths
- **Speckle noise**: Gaussian multiplicative noise

## Advanced Usage

### Per-Vertebra Force Analysis

Enable loss tracking for visualization:

```python
run_single_registration(
    ...,
    track_metrics=True,  # Enable metric logging
    save_transforms=True
)
```

Generates optimization plots showing:
- Mean similarity over iterations
- Axes penalty contribution
- IVD spacing loss contribution
- Total loss evolution

### Partial Ultrasound Data

Test robustness by removing portions of the US volume:

```bash
python extra/partial_two.py  # Blank out left or right side
python extra/occlusion.py     # Add random rectangular occlusions
```

## Related Repositories

- **[SOFA Spine Deformation](../SOFA_Spine_Deformation)**: Biomechanical simulation for generating ground-truth spinal deformations
