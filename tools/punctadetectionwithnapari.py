import tifffile as tiff
import numpy as np
from skimage.io import imsave
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import napari

viewer = napari.Viewer()

# Path to your 3-channel TIFF image
image_path = '/Users/amyzheng/Desktop/9994 VS Cell 1 Mt 550 at 915_V0_P_STACK.tif'
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

# Define threshold for gephyrin puncta
for num_stddevs in range(0, 1):
    threshold = mean_intensity + num_stddevs * std_intensity
    for min_puncta_size in range(4, 5):
        # Label the connected components for each plane separately
        for z_index in range(z):  # Iterate through each plane
            # Create a binary mask for the current plane
            puncta_mask_plane = normch4_dendrites[z_index] > threshold
            puncta_mask_plane = clear_border(puncta_mask_plane)
            
            # Label the connected components in the current plane
            labels_plane = label(puncta_mask_plane)
            
            # Create a binary mask for the filtered components in the current plane
            filtered_mask_plane = np.zeros_like(labels_plane, dtype=bool)
            for region in regionprops(labels_plane):
                if region.area >= min_puncta_size:  # Change to >= to count contiguous regions
                    # Add the region to the binary mask
                    filtered_mask_plane[labels_plane == region.label] = True
                    
                    # Extract contours for the current region
                    contours = find_contours(filtered_mask_plane, 0.5)  # 0.5 is the level to find contours
                    
                    # Add contours to the viewer as polygons
                    for contour in contours:
                        viewer.add_shapes(contour, shape_type='polygon', edge_color='red', name=f'Contour Plane={z_index}')

            # Add the binary mask to the viewer for the current plane
            viewer.add_labels(filtered_mask_plane.astype(int), name=f'Thresh={num_stddevs} connect={min_puncta_size} Plane={z_index}')

# Threshold normch4 to detect gephyrin puncta

# # Label and save puncta ROIs
# labels = label(puncta_mask)
# props = regionprops(labels)

# roi_save_path = 'puncta_rois.txt'
# with open(roi_save_path, 'w') as f:
#     for prop in props:
#         f.write(f"{prop.bbox}\n")

# print(f"ROIs saved to {roi_save_path}")

# Display with napari
viewer.add_image(image_with_normch4, name='Image with NormCh4')
napari.run()
