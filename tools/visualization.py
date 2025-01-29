import numpy as np
import napari
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

'''
12/2/24 - With analyzed vs not analyzed branches diff colors

Code produces 4 Layers:
1) Unanalyzed Branch Curves: Fit with spline unanalyzed branches
2) Analyzed Branch Curves: Fit with spline analyzed branches
3) Filtered XYZ Points: All of the branch points for the branches that have NOT been analyzed
4) Analyzed Branches: All of the branch points for the branches that have been analyzed
5) syntdpoints
6) inhpoints
7) XYZ Points: All of the branch points from the SNT Tracing.
Code also produces a matplot figure(similar to how Phoebe used to do it
but it is not updated with the different colors)

Press the eye next to the XYZ points to see the layers for the figure


To change colors - Go to the following lines and change the edge color and the face color
                 - You can use hex codes
Command F the title of the layer and change edge color and face color for it

Command F CHANGE SMOOTHING for smoothing parameter
'''


# Function to load and display image in Napari viewer
def load_and_display_image(viewer, image_path):
    image = tif.imread(image_path)
    ch1 = image[:, 0, :, :]
    ch2 = image[:, 1, :, :]
    ch3 = image[:, 2, :, :] 
    viewer.add_image(ch1, name='Gephyrin', colormap='green', blending='additive')
    viewer.add_image(ch2, name='RFP', colormap='cyan', blending='additive')
    viewer.add_image(ch3, name='Cell Fill', colormap='gray', blending='additive')

# Function to load tracing CSV and add points to the Napari viewer
def load_tracing_csv(viewer, csv_file_path):
    df = pd.read_csv(csv_file_path)
    coordinates = df[['x', 'y', 'z']].values
    viewer.add_points(coordinates, size=2, face_color='grey', edge_color='grey', name='XYZ Points')
    return coordinates

# Function to transform z-values and handle NaNs
def transform_z(z):
    if pd.isna(z):
        return np.nan
    return (int(z) - 1) / 4

# Function to apply scaling transformations
def apply_scaling_transformations(df, x_col, y_col, z_col, z_scaling):
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df[z_col] = pd.to_numeric(df[z_col], errors='coerce')
    df[z_col] = df[z_col].apply(transform_z)
    df = df.dropna(subset=[x_col, y_col, z_col])

    df[z_col] *= z_scaling
    return df[[x_col, y_col, z_col]].values

# Function to load Puncta ROIs and apply transformations
def load_and_transform_puncta_rois(csv_file_path, z_scaling):
    df = pd.read_csv(csv_file_path)
    syntd_subset = df[df['S 1'].isin(['ShaftwithSynTd', 'SpinewithSynTd'])]
    inh_subset = df[df['S 1'].isin(['InhibitoryShaft', 'SpinewithInhsynapse'])]
    
    syntd_positions = apply_scaling_transformations(syntd_subset, 'xpos S1', 'ypos S1', 'zpos S1', z_scaling)
    inh_positions = apply_scaling_transformations(inh_subset, 'xpos S1', 'ypos S1', 'zpos S1', z_scaling)
    
    return syntd_positions, inh_positions

# Function to add positions as points in Napari viewer
def add_positions_to_viewer(viewer, positions_1, positions_2):
    viewer.add_points(positions_1, size=5, face_color='red', edge_color='red', name='syntd Points', blending = 'additive')
    viewer.add_points(positions_2, size=5, face_color='blue', edge_color='blue', name='inh Points', blending = 'additive')

# Function to plot 3D scatter
def plot_3d_scatter(x_coords, y_coords, z_coords, xpos, ypos, zpos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='grey', marker='o', label='CSV Points', zorder=2)
    ax.scatter(xpos, ypos, zpos, c='r', marker='o', label='S1 Points', zorder=1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Scatter Plot of XYZ Positions')
    ax.legend()
    plt.show()

# Function to mark analyzed branches
def mark_analyzed_branches(coordinates, syntd_positions, inh_positions, branch_numbers, viewer):
    analyzed_branches = set()
    for i, branch_num in enumerate(branch_numbers):
        branch_coords = coordinates[branch_numbers == branch_num]
        distances = np.linalg.norm(branch_coords[:, None] - syntd_positions, axis=2)
        distances2 = np.linalg.norm(branch_coords[:, None] - inh_positions, axis=2)
        if np.any(distances < 10) or np.any(distances2 <10):  # Threshold of 5 pixel to be considered analyzed
            analyzed_branches.add(branch_num)
    
    # Highlight points of analyzed branches
    analyzed_coords = coordinates[np.isin(branch_numbers, list(analyzed_branches))]
    return analyzed_branches, analyzed_coords

# Update load_tracing_csv function to include branch numbers
def load_tracing_csv_with_branches(viewer, csv_file_path, z_scaling):
    df = pd.read_csv(csv_file_path)
    df['z'] *= z_scaling  # Apply z-scaling to the dataframe
    coordinates = df[['x', 'y', 'z']].values  # Extract scaled coordinates
    branch_numbers = df['path'].values  # Assuming a 'path' column for branches
    viewer.add_points(coordinates, size=3, face_color='blue', edge_color='blue', name='XYZ Points')
    return coordinates, branch_numbers

def exclude_analyzed_points(coordinates, branch_numbers, analyzed_branches):
    mask = ~np.isin(branch_numbers, list(analyzed_branches))
    filtered_coordinates = coordinates[mask]
    filtered_branch_numbers = branch_numbers[mask]
    return filtered_coordinates, filtered_branch_numbers

def interpolate_branch_curve(points, num_points=100):
    """
    Interpolate branch points into a smooth curve.
    Args:
        points (np.ndarray): Points to interpolate (Nx3).
        num_points (int): Number of points in the interpolated curve.
    Returns:
        np.ndarray: Interpolated curve points.
    """
    if len(points) < 4:
        print("Not enough points to interpolate a smooth curve.")
        return None

    try:
        #CHANGE SMOOTHING PARAMETERS HERE s = whatever
        tck, _ = splprep([points[:, 0], points[:, 1], points[:, 2]], s=40)
        #s here is smoothness
        u = np.linspace(0, 1, num_points)
        interpolated = np.array(splev(u, tck)).T  
        return interpolated
    except Exception as e:
        print(f"Error during spline interpolation: {e}")
        return None


def add_analyzed_and_unanalyzed_curves(viewer, coordinates, branch_numbers, syntd_positions, inh_positions):
    """
    Add interpolated curves for analyzed and unanalyzed branches as separate Napari layers.
    Args:
        viewer (napari.Viewer): The Napari viewer.
        coordinates (np.ndarray): Coordinates of points (Nx3).
        branch_numbers (np.ndarray): Branch identifiers for points (N,).
        syntd_positions (np.ndarray): Array of syntd positions (Mx3).
    """
    # Identify analyzed branches
    analyzed_branches, _ = mark_analyzed_branches(coordinates, syntd_positions, inh_positions, branch_numbers, viewer)

    # Combine coordinates and branch numbers into a DataFrame
    df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
    df['branch'] = branch_numbers

    # Initialize lists for analyzed and unanalyzed curves
    analyzed_curves = []
    unanalyzed_curves = []

    for branch_num in df['branch'].unique():
        branch_points = df[df['branch'] == branch_num][['x', 'y', 'z']].values

        # Interpolate the curve
        curve = interpolate_branch_curve(branch_points, num_points=100)
        if curve is not None:
            if branch_num in analyzed_branches:
                analyzed_curves.append(curve)
            else:
                unanalyzed_curves.append(curve)

    # Add analyzed curves as one layer
    if analyzed_curves:
        viewer.add_shapes(
            analyzed_curves,
            shape_type='path',
            edge_color='grey',
            edge_width=0.5,
            name='Analyzed Branch Curves', 
            blending = 'additive'
        )

    # Add unanalyzed curves as another layer
    if unanalyzed_curves:
        viewer.add_shapes(
            unanalyzed_curves,
            shape_type='path',
            edge_color='blue',
            edge_width=0.5,
            name='Unanalyzed Branch Curves', 
            blending = 'additive'
        )

def process_and_add_splines(file_path, viewer, num_points=100):
    """
    Processes a CSV file, interpolates branch data using splines, and adds the results to a napari viewer.

    Parameters:
        file_path (str): Path to the CSV file containing branch data.
        viewer (napari.Viewer): An instance of the napari viewer.
        num_points (int): Number of points to interpolate per branch. Default is 100.

    Returns:
        None
    """
    def interpolate_branch(data):
        """
        Interpolates a spline for the given data.

        Parameters:
            data (DataFrame): DataFrame containing 'X', 'Y', 'Z' columns for a branch.

        Returns:
            np.ndarray: Array of interpolated points (shape: num_points x 3).
        """
        z_scale = 4
        x, y, z = data['x'].values, data['y'].values, data['z'].values
        z= z * z_scale
        try:
            # Fit a spline to the data
            tck, u = splprep([x, y, z], s=0)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new, z_new = splev(u_new, tck)
            return np.stack([ x_new, y_new, z_new], axis=1)  # Return as an array
        except Exception as e:
            print(f"Interpolation failed for branch: {e}")
            return None

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Group by branch and interpolate
    all_curves = []
    for branch, group in df.groupby('path'):
        print(f"Processing branch: {branch}")
        interpolated = interpolate_branch(group)
        if interpolated is not None:
            all_curves.append(interpolated)  # Append the interpolated points

    # Add the fitted curves as a layer to the viewer
    if all_curves:
        viewer.add_shapes(
            all_curves,
            shape_type='path',
            edge_color='cyan',  # Customizable
            edge_width=0.5,
            name='Spline Fitted Curves',
            blending='additive',
            opacity=0.7  # Optional opacity
        )
        print("Fitted curves added to viewer.")
    else:
        print("No valid curves to add to the viewer.")

def main():
    print('running')
    viewer = napari.Viewer()

    z_scaling = 4
    
    tracing_csv_path ='/Volumes/nedividata/Amy/codeforjoe/SOM022Image1FullTrace_withbrancheslabeled_xyzCoordinates.csv'
    coordinates, branch_numbers = load_tracing_csv_with_branches(viewer, tracing_csv_path, z_scaling)
    # Exclude points in analyzed branches
    
    
    puncta_rois_path = '/Volumes/nedividata/Amy/codeforjoe/SOM022_PunctaROIs.csv'
    syntd_positions, inh_positions = load_and_transform_puncta_rois(puncta_rois_path, z_scaling)
    add_positions_to_viewer(viewer, syntd_positions, inh_positions)
    print('running2')
    # Mark analyzed branches
    
    analyzed_branches, analyzed_coords = mark_analyzed_branches(coordinates, syntd_positions, inh_positions, branch_numbers, viewer)
    # viewer.add_points(analyzed_coords, size=3, face_color='grey', edge_color='grey', name='Analyzed Branches')
    print(f"Analyzed branches: {analyzed_branches}")
    
    filtered_coordinates, filtered_branch_numbers = exclude_analyzed_points(coordinates, branch_numbers, analyzed_branches)
    # viewer.add_points(filtered_coordinates, size=3, face_color='green', edge_color='green', name='Filtered XYZ Points')

    print(f'coordinates:',coordinates)
    add_analyzed_and_unanalyzed_curves(viewer, coordinates, branch_numbers, syntd_positions, inh_positions)
    print('running3')

    # Plotting 3D scatter plot
    x_coords, y_coords, z_coords = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    xpos, ypos, zpos = syntd_positions[:, 0], syntd_positions[:, 1], syntd_positions[:, 2]
    plot_3d_scatter(x_coords, y_coords, z_coords, xpos, ypos, zpos)

    # Start the Napari event loop
    napari.run()


# Run the main function
if __name__ == "__main__":
    main()
