from skimage.morphology import skeletonize_3d, remove_small_objects, skeletonize
from skimage.filters import threshold_otsu
import numpy as np
from scipy.spatial import KDTree
import napari
import tifffile as tf

def extract_2d_points(image_3d, method="binary", min_size=50):
    """
    Extract points from a 3D image projected into 2D.
    Parameters:
    - image_3d: 3D numpy array.
    - method: 'binary' or 'skeleton'.
    - min_size: Minimum size of objects to retain when skeletonizing.
    """
    # Max projection along the Z-axis
    image_2d = np.max(image_3d, axis=0)

    # Threshold
    binary = image_2d > threshold_otsu(image_2d)

    if method == "binary":
        # Use all points in the binary mask
        points = np.column_stack(np.where(binary))
    elif method == "skeleton":
        # Skeletonize with small object removal
        binary = remove_small_objects(binary, min_size=min_size)
        skeleton = skeletonize(binary)
        points = np.column_stack(np.where(skeleton))
    else:
        raise ValueError("Invalid method. Choose 'binary' or 'skeleton'.")

    return points  # Ensure float type for scaling

def extract_2d_points_direct(image_2d, method="binary", min_size=50):
    """
    Extract points from a 2D image.
    Parameters:
    - image_2d: 2D numpy array.
    - method: 'binary' or 'skeleton'.
    - min_size: Minimum size of objects to retain when skeletonizing.
    """
    # Threshold
    binary = image_2d > threshold_otsu(image_2d)+30

    if method == "binary":
        # Use all points in the binary mask
        points = np.column_stack(np.where(binary))
        binary = remove_small_objects(binary, min_size=min_size)
        skeleton = skeletonize(binary)
        points = np.column_stack(np.where(skeleton))
    else:
        raise ValueError("Invalid method. Choose 'binary' or 'skeleton'.")

    return points.astype(float)  # Ensure float type for scaling
# def extract_3d_skeleton_points(image_3d):
#     # Convert 3D image to binary
#     binary = image_3d > threshold_otsu(image_3d)
#     skeleton = skimage.morphology.skeletonize(binary)
#     # Get coordinates of skeleton points
#     points = np.column_stack(np.where(skeleton))
#     return points

def extract_2d_skeleton_points(image_3d):
    """
    Project 3D image into 2D using max projection along the Z-axis,
    then extract skeleton points from the resulting 2D image.
    """
    # Max projection along the Z-axis
    image_2d = np.max(image_3d, axis=0)

    # Threshold and skeletonize
    binary = image_2d > threshold_otsu(image_2d)
    skeleton = skeletonize_3d(binary)

    # Get coordinates of skeleton points
    points = np.column_stack(np.where(skeleton))
    return points


def extract_2d_points(image_2d):
    """
    Extract skeleton points directly from a 2D image.
    """
    binary = image_2d > threshold_otsu(image_2d)
    skeleton = skeletonize_3d(binary)
    points = np.column_stack(np.where(skeleton))
    return points

def icp_with_scaling(source, target, max_iterations=50, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) with scaling alignment.
    Parameters:
    source (ndarray): Source point cloud (N x D).
    target (ndarray): Target point cloud (M x D).
    max_iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.
    Returns:
    aligned_source (ndarray): Transformed source point cloud.
    transformation (dict): Final transformation matrix, translation vector, and scaling factor.
    """
    # Initialize the source point cloud
    src = source.copy()

    # Initialize transformation matrix and scaling
    transformation_matrix = np.eye(src.shape[1])
    translation_vector = np.zeros(src.shape[1])
    scaling_factor = 1.0

    for i in range(max_iterations):
    # Find closest points in the target for each source point
        tree = KDTree(target)
        distances, indices = tree.query(src)

        # Compute centroids of source and target
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(target[indices], axis=0)

        # Normalize points by centroids
        src_centered = src - centroid_src
        tgt_centered = target[indices] - centroid_tgt

        # Estimate scaling factor
        src_norm = np.linalg.norm(src_centered, axis=1)
        tgt_norm = np.linalg.norm(tgt_centered, axis=1)
        scaling_factor = np.mean(tgt_norm / src_norm)

        # Scale the source points
        src_scaled = src_centered * scaling_factor

        # Compute covariance matrix
        H = src_scaled.T @ tgt_centered

        # Compute Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T # Rotation matrix

        # Correct for reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation vector
        t = centroid_tgt - scaling_factor * R @ centroid_src

        # Update the source point cloud
        src = (scaling_factor * R @ src.T).T + t

        # Update cumulative transformation
        transformation_matrix = scaling_factor * R @ transformation_matrix
        translation_vector = scaling_factor * R @ translation_vector + t

        # Check convergence
        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    # Return the aligned source and the transformation
    return src, {
    "rotation": transformation_matrix,
    "translation": translation_vector,
    "scaling": scaling_factor,
    }

# # Load images
# small_image = tf.imread(r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1F3withoutStreaks.tif")
# large_skeleton_image_2d = tf.imread(r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\MAX_2024-8-30_1317_dob2-16-24_session2.lif - Series001.tif")

# # Extract channels from the small image
# _, _, small_skeleton_image_3d, _ = small_image[:, 0, :, :], small_image[:, 1, :, :], small_image[:, 2, :, :], small_image[:, 3, :, :]

# # Extract skeleton points
# small_skeleton_points_2d = extract_2d_skeleton_points(small_skeleton_image_3d)
# large_skeleton_points_2d = extract_2d_points(large_skeleton_image_2d)

# # Scale the 2D small skeleton points
# scaling_factors = np.array([5, 5])  # Scaling in X and Y
# large_skeleton_points_2d *= scaling_factors

# # Perform ICP with scaling alignment
# aligned_points_2d, transformation = icp_with_scaling(small_skeleton_points_2d, large_skeleton_points_2d)

# # Visualize in Napari
# viewer = napari.Viewer(ndisplay=2)



# Load images
small_image = tf.imread(r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1F3withoutStreaks.tif")
large_skeleton_image_2d = tf.imread(r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\MAX_2024-8-30_1317_dob2-16-24_session2.lif - Series001.tif")
# large_skeleton_image_2d = np.max(large_skeleton_image_2d) - large_skeleton_image_2d

# Extract channels from the small image
_, _, small_skeleton_image_3d, _ = small_image[:, 0, :, :], small_image[:, 1, :, :], small_image[:, 2, :, :], small_image[:, 3, :, :]

# Extract points
small_skeleton_points_2d = extract_2d_points(small_skeleton_image_3d)  # Use 'binary' or 'skeleton'
large_skeleton_points_2d = extract_2d_points_direct(large_skeleton_image_2d)

# Scale the 2D small skeleton points
# Ensure the points array has two dimensions (X, Y)
small_skeleton_points_2d = small_skeleton_points_2d[:, :2]

# Apply scaling factors
scaling_factors = np.array([5, 5])  # Scaling in X and Y
large_skeleton_points_2d *= scaling_factors
# Visualize in Napari
viewer = napari.Viewer(ndisplay=2)

# Add large skeleton
viewer.add_points(
    large_skeleton_points_2d,
    size=5,
    face_color="blue",
    name="Large Skeleton"
)

# Add scaled small skeleton
viewer.add_points(
    small_skeleton_points_2d,
    size=5,
    face_color="red",
    name="Small Skeleton (Scaled)"
)
napari.run()
