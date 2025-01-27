import tifffile as tiff
import numpy as np
from skimage.io import imsave
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import napari
from scipy.spatial import ConvexHull


viewer = napari.Viewer()

# Path to your 3-channel TIFF image
image_path = r"Z:\Amy\files_for_amy_fromJoe\example_analysis_SOM022\Automated_Puncta_Detection\Image1\With_normch4-SOM022_Image 1_MotionCorrected.tif"
# Load the image
image = tiff.imread(image_path)
print(image.shape)

# image_copy1 = np.copy(image)
# image_copy2 = np.copy(image)
# image_copy3 = np.copy(image)
# # except Exception as e:
# #     print(f"Error loading image: {e}")
# #     exit()

# # print(image.shape)    
# viewer.add_image(image)

z, channels, x, y = image.shape
ch1 = image[:, 0, :, :]
ch2 = image[:, 1, :, :]
ch3 = image[:, 2, :, :]


print(ch1.shape)
print(ch2.shape)
print(ch3.shape)

# # Save individual channels

viewer.add_image(ch1, name = 'Ch1: Cell Fill')
viewer.add_image(ch2, name = 'Ch2: Gephyrin')
viewer.add_image(ch3)


# # Split the channels assuming each channel is a different layer in the Z-stack
# ch1 = image[0]
# ch2 = image[1]
# ch3 = image[2]

# viewer.add_image(ch1)
# viewer.add_image(ch2)
# viewer.add_image(ch3)

# # Save individual channels
# imsave('ch1.tif', ch1)
# imsave('ch2.tif', ch2)
# imsave('ch3.tif', ch3)

# Normalize channel 2 using channel 1
ch2multiplied = ch2*100
viewer.add_image(ch2multiplied)
normch4 = ch2multiplied/ch1
viewer.add_image(normch4, contrast_limits =(0, 80) )

# # Save normalized channel
# imsave('normch4.tif', normch4.astype(np.uint8))

# Define the brightness range for dendrites in channel 1
dendritemin = 10
dendritemax = 80

# Create a mask for dendrites based on brightness range in ch1
dendrite_mask = (ch1 >= dendritemin) & (ch1 <= dendritemax)

# Apply the mask to normch4
normch4_dendrites = np.where(dendrite_mask, normch4, 0)
viewer.add_image(normch4_dendrites)
# imsave('normch4_dendrites.tif', normch4_dendrites.astype(np.uint8))

# Create a 4-channel image
image_with_normch4 = np.stack([ch1, ch2, ch3, normch4_dendrites], axis=0)
# imsave('image_with_normch4.tif', image_with_normch4.astype(np.uint8))

# Calculate mean and standard deviation of normch4 in dendrite areas
dendrite_pixels = normch4_dendrites[dendrite_mask]
mean_intensity = dendrite_pixels.mean()
std_intensity = dendrite_pixels.std()

print(f"Mean intensity: {mean_intensity}")
print(f"Standard deviation: {std_intensity}")

# Initialize a 3D array for stacking filtered mask planes
stacked_labels = np.zeros((z, x, y), dtype=int)

# Create a shapes layer for polygons
shapes_layer = viewer.add_shapes(
    name="Detected Regions", 
    edge_color="red", 
    face_color="transparent", 
    shape_type="polygon"
)

# Loop over threshold and minimum puncta size combinations
for num_stddevs in range(2, 3):
    threshold = mean_intensity + num_stddevs * std_intensity
    for min_puncta_size in range(4, 5):
        # Process each z-plane separately
        for z_index in range(z):
            # Create a binary mask for the current plane
            puncta_mask_plane = normch4_dendrites[z_index] > threshold
            puncta_mask_plane = clear_border(puncta_mask_plane)
            
            # Label the connected components in the current plane
            labels_plane = label(puncta_mask_plane)
            
            # Filter regions and add polygons
            for region in regionprops(labels_plane):
                if region.area >= min_puncta_size:
                    coords = region.coords
                    coords_2d = [(coord[0], coord[1]) for coord in coords]

                    # Check if there are enough unique coordinates in 2D
                    if len(coords_2d) >= 3 and len(set([coord[0] for coord in coords_2d])) > 1 and len(set([coord[1] for coord in coords_2d])) > 1:
                        # Only compute convex hull if points span 2D
                        hull = ConvexHull(coords_2d)
                        # Reorder coordinates in the correct order for the polygon
                        ordered_coords = [[z_index, coord[0], coord[1]] for i in hull.vertices for coord in [coords_2d[i]]]
                        # Add the ordered contour as a polygon layer in Napari
                        shapes_layer.add(
                            ordered_coords, 
                            shape_type="polygon", 
                            edge_width=0.1
                        )
                    # Add the region to the stacked label
                    stacked_labels[z_index][labels_plane == region.label] = 1

# Add the normalized 4-channel image for reference
viewer.add_image(image_with_normch4, name='Image with NormCh4')
napari.run()
