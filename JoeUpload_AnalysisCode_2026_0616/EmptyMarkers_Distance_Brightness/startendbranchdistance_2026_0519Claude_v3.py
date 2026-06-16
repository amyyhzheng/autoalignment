import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# GET INPUT DIRECTORIES
# ============================================================================
print("=" * 70)
print("BRANCH DISTANCE CALCULATOR")
print("=" * 70)

puncta_scoring_dir = input("\nEnter the PunctaScoring directory path: ").strip()
puncta_scoring_dir = Path(puncta_scoring_dir)

if not puncta_scoring_dir.exists():
    raise ValueError(f"Directory does not exist: {puncta_scoring_dir}")

trace_file_path = input("Enter the full path to the trace file (e.g. SOM055_Image5_FullTrace_xyzCoordinates.csv): ").strip()
trace_file_path = Path(trace_file_path)

if not trace_file_path.exists():
    raise ValueError(f"Trace file does not exist: {trace_file_path}")

print(f"\nPunctaScoring directory: {puncta_scoring_dir}")
print(f"Trace file: {trace_file_path}")

# ============================================================================
# LOAD TRACE FILE ONCE (contains all branches)
# ============================================================================
print("\nLoading trace file...")
trace_df = pd.read_csv(trace_file_path)
print(f"Trace file loaded: {len(trace_df)} total rows")

# ============================================================================
# PROCESS EACH BRANCH
# ============================================================================
BRANCH_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []

for branch_id in BRANCH_IDS:
    print("\n" + "=" * 70)
    print(f"PROCESSING BRANCH {branch_id}")
    print("=" * 70)
    
    # Construct path to branch marker file
    branch_dir = puncta_scoring_dir / f"branch{branch_id}"
    #ManuallyEdit
    branch_marker_file = branch_dir / f"SOM056_Image4_branch{branch_id}.csv"
    
    # Check if file exists
    if not branch_marker_file.exists():
        print(f"  WARNING: File not found: {branch_marker_file}")
        print(f"  Skipping branch {branch_id}")
        continue
    
    try:
        # ====================================================================
        # LOAD BRANCH MARKER FILE
        # ====================================================================
        print(f"Loading marker file: {branch_marker_file}")
        branch_df = pd.read_csv(branch_marker_file)
        
        # ====================================================================
        # EXTRACT START AND END BRANCH MARKERS
        # ====================================================================
        print("Finding StartBranch and EndBranch markers...")
        
        start_row = branch_df[branch_df['type'].str.lower() == 'startbranch']
        end_row = branch_df[branch_df['type'].str.lower() == 'endbranch']
        
        if len(start_row) == 0 or len(end_row) == 0:
            print(f"  WARNING: Could not find StartBranch or EndBranch markers")
            print(f"  Skipping branch {branch_id}")
            continue
        
        # Extract coordinates from markers (axis-0=z, axis-1=y, axis-2=x)
        # Scale: axis-2 * 0.25 (X), axis-1 * 0.25 (Y), axis-0 * 0.25 (Z)
        start_x_um = start_row['axis-2'].values[0] * 0.25
        start_y_um = start_row['axis-1'].values[0] * 0.25
        start_z_um = start_row['axis-0'].values[0] * 0.25
        
        end_x_um = end_row['axis-2'].values[0] * 0.25
        end_y_um = end_row['axis-1'].values[0] * 0.25
        end_z_um = end_row['axis-0'].values[0] * 0.25
        
        start_marker_um = np.array([start_x_um, start_y_um, start_z_um])
        end_marker_um = np.array([end_x_um, end_y_um, end_z_um])
        
        print(f"  StartBranch (µm): x={start_x_um:.2f}, y={start_y_um:.2f}, z={start_z_um:.2f}")
        print(f"  EndBranch (µm):   x={end_x_um:.2f}, y={end_y_um:.2f}, z={end_z_um:.2f}")
        
        # ====================================================================
        # FILTER TRACE DATA FOR THIS BRANCH
        # ====================================================================
        print(f"Filtering trace data for branch{branch_id}...")
        
        trace_branch = trace_df[trace_df['path'].str.lower() == f'branch{branch_id}'.lower()].copy()
        
        if len(trace_branch) == 0:
            print(f"  WARNING: No trace data found for branch{branch_id}")
            print(f"  Skipping branch {branch_id}")
            continue
        
        # Get x, y, z coordinates in original pixel units
        trace_coords_pixels = trace_branch[['x', 'y', 'z']].values.astype(float)
        
        print(f"  Found {len(trace_coords_pixels)} trace points for branch{branch_id}")
        
        # ====================================================================
        # SCALE TRACE COORDINATES FOR DISTANCE CALCULATION
        # ====================================================================
        # Scale: x * 0.25, y * 0.25, z * 1.0
        trace_coords_scaled = trace_coords_pixels.copy()
        trace_coords_scaled[:, 0] *= 0.25  # x
        trace_coords_scaled[:, 1] *= 0.25  # y
        trace_coords_scaled[:, 2] *= 1.0   # z (no scaling)
        
        # ====================================================================
        # FIND CLOSEST POINTS TO START AND END MARKERS
        # ====================================================================
        print("Finding closest trace points to markers...")
        
        # Calculate distances from each trace point to the start marker
        distances_to_start = np.linalg.norm(trace_coords_scaled - start_marker_um, axis=1)
        start_idx = np.argmin(distances_to_start)
        start_dist_to_marker = distances_to_start[start_idx]
        
        # Calculate distances from each trace point to the end marker
        distances_to_end = np.linalg.norm(trace_coords_scaled - end_marker_um, axis=1)
        end_idx = np.argmin(distances_to_end)
        end_dist_to_marker = distances_to_end[end_idx]
        
        print(f"  Start marker closest to trace point {start_idx} (distance: {start_dist_to_marker:.3f} µm)")
        print(f"  End marker closest to trace point {end_idx} (distance: {end_dist_to_marker:.3f} µm)")
        
        # ====================================================================
        # ENSURE START COMES BEFORE END IN THE TRACE
        # ====================================================================
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            print(f"  Swapped indices: start_idx={start_idx}, end_idx={end_idx}")
        
        # ====================================================================
        # EXTRACT SEGMENT BETWEEN START AND END
        # ====================================================================
        print(f"Extracting segment from index {start_idx} to {end_idx}...")
        
        segment_pixels = trace_coords_pixels[start_idx:end_idx+1]
        print(f"  Segment has {len(segment_pixels)} points")
        
        # ====================================================================
        # SCALE SEGMENT COORDINATES TO MICRONS
        # ====================================================================
        print("Scaling segment to microns (x*0.25, y*0.25, z*1.0)...")
        
        segment_scaled = segment_pixels.astype(float).copy()
        segment_scaled[:, 0] *= 0.25  # x
        segment_scaled[:, 1] *= 0.25  # y
        segment_scaled[:, 2] *= 1.0   # z
        
        print(f"  First point: ({segment_scaled[0, 0]:.2f}, {segment_scaled[0, 1]:.2f}, {segment_scaled[0, 2]:.2f}) µm")
        print(f"  Last point: ({segment_scaled[-1, 0]:.2f}, {segment_scaled[-1, 1]:.2f}, {segment_scaled[-1, 2]:.2f}) µm")
        
        # ====================================================================
        # CALCULATE DISTANCE ALONG THE SEGMENT
        # ====================================================================
        print("Calculating distance between consecutive points...")
        
        total_distance = 0.0
        distances = []
        
        for i in range(len(segment_scaled) - 1):
            p1 = segment_scaled[i]
            p2 = segment_scaled[i + 1]
            
            # Euclidean distance
            dist = np.linalg.norm(p2 - p1)
            distances.append(dist)
            total_distance += dist
        
        print(f"  Number of segments: {len(distances)}")
        print(f"  Min segment distance: {np.min(distances):.4f} µm")
        print(f"  Max segment distance: {np.max(distances):.4f} µm")
        print(f"  Mean segment distance: {np.mean(distances):.4f} µm")
        
        # ====================================================================
        # STORE RESULT
        # ====================================================================
        print(f"\n>>> BRANCH {branch_id}: {total_distance:.4f} µm")
        
        results.append({
            'branch': branch_id,
            'distance_um': total_distance,
            'num_points': len(segment_scaled),
            'start_marker_x_um': start_x_um,
            'start_marker_y_um': start_y_um,
            'start_marker_z_um': start_z_um,
            'end_marker_x_um': end_x_um,
            'end_marker_y_um': end_y_um,
            'end_marker_z_um': end_z_um,
        })
        
    except Exception as e:
        print(f"  ERROR processing branch {branch_id}: {e}")
        continue

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL BRANCHES")
print("=" * 70)

if len(results) > 0:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results to CSV in PunctaScoring directory
    output_file = puncta_scoring_dir / "Length_Analyzed_Segments.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
else:
    print("No results to report")

print("\n" + "=" * 70)