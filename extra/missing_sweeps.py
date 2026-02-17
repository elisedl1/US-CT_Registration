"""
simulate missing sweeps in ultrasound volumes for reconstruction testing

1. random dropout: up to 25% of slices blanked out
2. tracking failures: 5-10 consecutive frames missing
3. random dropout: 2-5 consecutive frames missing
"""

import numpy as np
import nrrd
import os
import random


def blank_slice(volume, slice_idx, axis=2):
    if axis == 0:
        volume[slice_idx, :, :] = 0
    elif axis == 1:
        volume[:, slice_idx, :] = 0
    else:  # axis == 2
        volume[:, :, slice_idx] = 0


def simulate_random_dropout(volume, max_dropout_percent, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    volume_copy = volume.copy()
    n_slices = volume.shape[2]
    
    # randomly choose how many slices to remove (up to max_dropout_percent)
    n_to_remove = np.random.randint(1, int(n_slices * max_dropout_percent) + 1)
    
    # randomly select which slices to remove
    slices_to_remove = np.random.choice(n_slices, size=n_to_remove, replace=False)
    slices_to_remove = sorted(slices_to_remove)
    
    # blank out selected slices
    for slice_idx in slices_to_remove:
        blank_slice(volume_copy, slice_idx)
    
    return volume_copy, slices_to_remove


def simulate_tracking_failures(volume, n_failures=None, consecutive_range=(5, 10), seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    volume_copy = volume.copy()
    n_slices = volume.shape[2]
    
    # random number of failure events if not specified
    if n_failures is None:
        n_failures = random.randint(2, 5)
    
    failures = []
    
    for _ in range(n_failures):
        # random consecutive length
        n_consecutive = random.randint(consecutive_range[0], consecutive_range[1])
        
        # random starting position (ensure we don't go past the end)
        max_start = n_slices - n_consecutive
        if max_start <= 0:
            continue
        
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + n_consecutive
        
        # check for overlap with existing failures
        overlap = False
        for existing_start, existing_end in failures:
            if not (end_idx <= existing_start or start_idx >= existing_end):
                overlap = True
                break
        
        if not overlap:
            # blank out consecutive slices
            for slice_idx in range(start_idx, end_idx):
                blank_slice(volume_copy, slice_idx)
            
            failures.append((start_idx, end_idx))
    
    return volume_copy, failures


def simulate_random_burst_dropout(volume, n_bursts=None, consecutive_range=(2, 5), seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    volume_copy = volume.copy()
    n_slices = volume.shape[2]
    
    # random number of burst events if not specified
    if n_bursts is None:
        n_bursts = random.randint(5, 15)
    
    bursts = []
    
    for _ in range(n_bursts):
        # random consecutive length
        n_consecutive = random.randint(consecutive_range[0], consecutive_range[1])
        
        # random starting position
        max_start = n_slices - n_consecutive
        if max_start <= 0:
            continue
        
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + n_consecutive
        
        # check for overlap
        overlap = False
        for existing_start, existing_end in bursts:
            if not (end_idx <= existing_start or start_idx >= existing_end):
                overlap = True
                break
        
        if not overlap:
            # blank out consecutive slices
            for slice_idx in range(start_idx, end_idx):
                blank_slice(volume_copy, slice_idx)
            
            bursts.append((start_idx, end_idx))
    
    return volume_copy, bursts


def simulate_combined_dropout(volume, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    volume_copy = volume.copy()
    
    # apply tracking failures (5-10 consecutive)
    volume_copy, tracking_failures = simulate_tracking_failures(
        volume_copy, n_failures=random.randint(2, 4), 
        consecutive_range=(5, 8), seed=seed
    )
    
    # apply random burst dropout (2-5 consecutive)
    volume_copy, random_bursts = simulate_random_burst_dropout(
        volume_copy, n_bursts=random.randint(8, 15),
        consecutive_range=(2, 5), seed=seed+1 if seed else None
    )
    
    info = {
        'tracking_failures': tracking_failures,
        'random_bursts': random_bursts
    }
    
    return volume_copy, info


def main():

    input_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/Known_Trans/intra1/Cases/US_complete_cal.nrrd"
    
    print(f"Reading volume from: {input_path}")
    volume, header = nrrd.read(input_path)
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    
    output_dir = os.path.dirname(input_path)
    
    # base_seed = 42
    
    # 1. random dropout
    print("\n1. Creating random dropout version (up to 25% slices removed)...")
    volume_random, removed_slices = simulate_random_dropout(volume, max_dropout_percent=0.20, seed = None)
    output_path = os.path.join(output_dir, "US_missing_random_25pct.nrrd")
    nrrd.write(output_path, volume_random, header)
    print(f"   Saved to: {output_path}")
    print(f"   Removed {len(removed_slices)} slices ({len(removed_slices)/volume.shape[2]*100:.1f}%)")
    
    # 2. tracking failures (5-10 consecutive frames)
    print("\n2. Creating tracking failure version (5-10 consecutive frames)...")
    volume_tracking, tracking_failures = simulate_tracking_failures(
        volume, n_failures=2, consecutive_range=(5, 8), seed=None
    )
    output_path = os.path.join(output_dir, "US_missing_tracking_failures.nrrd")
    nrrd.write(output_path, volume_tracking, header)
    print(f"   Saved to: {output_path}")
    print(f"   Created {len(tracking_failures)} tracking failure events:")
    for i, (start, end) in enumerate(tracking_failures):
        print(f"     Failure {i+1}: slices {start}-{end-1} ({end-start} frames)")
    
    # 3. random burst dropout (2-5 consecutive frames)
    print("\n3. Creating random burst dropout version (2-5 consecutive frames)...")
    volume_bursts, bursts = simulate_random_burst_dropout(
        volume, n_bursts=8, consecutive_range=(2, 3), seed=None
    )
    output_path = os.path.join(output_dir, "US_missing_random_bursts.nrrd")
    nrrd.write(output_path, volume_bursts, header)
    print(f"   Saved to: {output_path}")
    print(f"   Created {len(bursts)} random burst events")
    
    # 4. combined dropout (tracking failures + random bursts)
    print("\n4. Creating combined dropout version (tracking + bursts)...")
    volume_combined, combined_info = simulate_combined_dropout(volume)
    output_path = os.path.join(output_dir, "US_missing_combined.nrrd")
    nrrd.write(output_path, volume_combined, header)
    print(f"   Saved to: {output_path}")
    print(f"   Tracking failures: {len(combined_info['tracking_failures'])} events")
    print(f"   Random bursts: {len(combined_info['random_bursts'])} events")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "missing_sweep_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Missing Sweep Simulation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original volume: {input_path}\n")
        f.write(f"Volume shape: {volume.shape}\n")
        f.write(f"Total slices: {volume.shape[2]}\n\n")
        
        f.write("1. Random Dropout (up to 25%):\n")
        f.write(f"   File: US_missing_random_25pct.nrrd\n")
        f.write(f"   Removed {len(removed_slices)} slices ({len(removed_slices)/volume.shape[2]*100:.1f}%)\n")
        f.write(f"   Slice indices: {removed_slices}\n\n")
        
        f.write("2. Tracking Failures (5-10 consecutive):\n")
        f.write(f"   File: US_missing_tracking_failures.nrrd\n")
        f.write(f"   Number of events: {len(tracking_failures)}\n")
        for i, (start, end) in enumerate(tracking_failures):
            f.write(f"   Event {i+1}: slices {start}-{end-1} ({end-start} frames)\n")
        f.write("\n")
        
        f.write("3. Random Bursts (2-5 consecutive):\n")
        f.write(f"   File: US_missing_random_bursts.nrrd\n")
        f.write(f"   Number of events: {len(bursts)}\n")
        for i, (start, end) in enumerate(bursts):
            f.write(f"   Event {i+1}: slices {start}-{end-1} ({end-start} frames)\n")
        f.write("\n")
        
        f.write("4. Combined (tracking + bursts):\n")
        f.write(f"   File: US_missing_combined.nrrd\n")
        f.write(f"   Tracking failures: {len(combined_info['tracking_failures'])} events\n")
        f.write(f"   Random bursts: {len(combined_info['random_bursts'])} events\n")
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()