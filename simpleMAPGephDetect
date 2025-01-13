import napari
import numpy as np
from skimage import io, filters, feature, measure
from scipy.ndimage import gaussian_filter
import tifffile as tf

image_path =r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Ch-GFP-568_Gp-Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1C1.tif"
image = tf.imread(image_path)
ch1 = image[:, 0, :, :]
# Step 2: Preprocess and find local maxima
# Apply a Gaussian filter for noise reduction (optional)
smoothed_image = gaussian_filter(ch1, sigma=2)

# Enhanced preprocessing and detection
def detect_gephyrin_structures(image_3d, sigma=2, min_size=5, max_size=20, threshold_rel=0.4):
    """Detect Gephyrin structures using multiple methods"""
    # Apply 3D Gaussian smoothing
    smoothed = gaussian_filter(image_3d, sigma=sigma)
    
    # Use more aggressive local thresholding
    thresh = filters.threshold_local(smoothed, block_size=21, offset=0.1)
    binary = smoothed > thresh
    
    # Additional intensity threshold to remove weak signals
    global_thresh = filters.threshold_otsu(smoothed)
    binary = binary & (smoothed > global_thresh)
    
    # Label connected components in 3D
    labels = measure.label(binary)
    props = measure.regionprops(labels, intensity_image=smoothed)  # Added intensity_image
    
    # Filter regions based on size, shape, and intensity
    valid_regions = []
    for prop in props:
        if (min_size <= prop.equivalent_diameter <= max_size and
            prop.mean_intensity > 1.5 * global_thresh):  # Added intensity check
            
            eigvals = prop.inertia_tensor_eigvals
            aspect_ratio = np.sqrt(max(eigvals) / min(eigvals)) if min(eigvals) > 0 else float('inf')
            
            # More stringent aspect ratio threshold
            if aspect_ratio < 2.5:
                valid_regions.append(prop.centroid)
    
    return np.array(valid_regions), binary

# Apply detection
coordinates, binary_mask = detect_gephyrin_structures(ch1)

# Create simple threshold mask for comparison
simple_threshold = ch1 > filters.threshold_otsu(ch1)

# Visualization with both methods
viewer = napari.Viewer()
viewer.add_image(ch1, name="Original")
viewer.add_image(binary_mask, name="Complex Detection Mask", opacity=0.5)
viewer.add_image(simple_threshold, name="Simple Threshold Mask", opacity=0.5, colormap='magenta')
viewer.add_points(coordinates, size=5, face_color='red', name="Detected Structures")

napari.run()
