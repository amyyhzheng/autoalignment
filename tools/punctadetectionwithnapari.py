import tifffile as tiff
import numpy as np
from skimage.io import imsave
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import napari
from scipy.spatial import ConvexHull
from skimage.morphology import binary_erosion, dilation, square


''''
Run with python 3.10.11 interpreter (control shift P - Python:Select Interpreter)

'''
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

# Function to sort points clockwise or counterclockwise
def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on the angles
    sorted_indices = np.argsort(angles)
    
    # Return the sorted points
    return points[sorted_indices]

# def get_polygon_outline(coords):
#     if not coords:
#         return []
    
#     try:
#         # Convert the input list of coordinates to a set for efficient lookup
#         coord_set = set(tuple(c) for c in coords)
        
#         # Directions for checking neighbors: right, down, left, up
#         directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
#         outline = []
        
#         # Step 1: Find boundary edges
#         for x, y in coords:
#             num_neighbors = []
#             for dx, dy in directions:
#                 nx, ny = x + dx, y + dy
#                 if (nx, ny) not in coord_set:
#                     # If the neighbor is not part of the shape (i.e., it's a 0), calculate the midpoint
#                     mid_x = (x + nx) / 2.0
#                     mid_y = (y + ny) / 2.0
#                     outline.append([mid_x, mid_y])
#                     num_neighbors.append([nx, ny])
#             if len(num_neighbors) == 2:
#                 nx_sum = 0
#                 ny_sum = 0
#                 for neighbor in num_neighbors:
#                     nx_sum += neighbor[0]
#                     ny_sum += neighbor[1]
#                 outline.append([nx_sum / len(num_neighbors), ny_sum / len(num_neighbors)])

#             # elif len(num_neighbors) == 3:
#             #     for i in range(len(num_neighbors)):
#             #         for j in range(i+1, len(num_neighbors)):
#             #             dx1, dy1 = num_neighbors
#             #     across_neighbors = []
#             #     for i in range(len(num_neighbors)):
#             #         for j in range(i + 1, len(num_neighbors)):
#             #             dx1, dy1 = num_neighbors[i][0] - x, num_neighbors[i][1] - y
#             #             dx2, dy2 = num_neighbors[j][0] - x, num_neighbors[j][1] - y
                        
#             #             if dx1 != -dx2 or dy1 != -dy2:
#             #                 # If they are not opposite, calculate the average of the two neighbors
#             #                 avg_nx = (num_neighbors[i][0] + num_neighbors[j][0]) / 2.0
#             #                 avg_ny = (num_neighbors[i][1] + num_neighbors[j][1]) / 2.0
#             #                 outline.append([avg_nx, avg_ny])
        
#         # Step 2: Sort points in a clockwise or counter-clockwise order
#         if outline:
#             outline = sorted(outline, key=lambda p: np.arctan2(p[1] - np.mean([pt[1] for pt in outline]), p[0] - np.mean([pt[0] for pt in outline])))
        
#         return outline
    
#     except Exception as e:
#         print(f"Error: {e}")
#         return []


def get_expanded_bounding_box(region):
    minr, minc, maxr, maxc = region.bbox
    return (minr - 1, minc - 1, maxr + 1, maxc + 1)

def get_all_half_grid_points(region):
    """Generates all 0.5 offset points inside the expanded bounding box."""
    minr, minc, maxr, maxc = get_expanded_bounding_box(region)

    # Create a grid of 0.5-aligned points
    r_vals = np.arange(minr - 0.5, maxr + 1, 1)
    c_vals = np.arange(minc - 0.5, maxc + 1, 1)

    # Generate all (row, col) pairs
    corners = [(r, c) for r in r_vals for c in c_vals]
    
    return corners

def filter_corners(region, binary_mask, coords):
    minr, minc, maxr, maxc = get_expanded_bounding_box(region)
    corners = get_corners(region)
    filtered_corners = []
    
    for r, c in corners:
        if not ((r, c) in coords and binary_mask[int(r), int(c)] and binary_mask[int(r-1), int(c-1)]):
            filtered_corners.append((r, c))
    
    return filtered_corners

# Loop over threshold and minimum puncta size combinations
for num_stddevs in range(1, 2):
    threshold = mean_intensity + num_stddevs * std_intensity
    for min_puncta_size in range(3, 5):
        shapes_layer = viewer.add_shapes(
            name=f'Thresh={num_stddevs}_MinSize={min_puncta_size}', 
            edge_color="red", 
            face_color="transparent", 
            shape_type="polygon"
        )
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
            # Create a binary mask of the current region
                    region_mask = labels_plane == region.label

                    # # Perform erosion to shrink the region slightly
                    # eroded_mask = binary_erosion(region_mask)

                    # # Detect boundaries: subtract eroded mask from the original mask to get the boundary
                    # boundary_mask = region_mask & ~eroded_mask
                    # # 1. Dilate the boundary mask by 1 pixel (expand the area outward)
                    # dilated_mask = dilation(region_mask)
                    # outside_boundary_mask = dilated_mask & ~region_mask

                    # 2. Subtract the original boundary mask from the dilated mask to get the boundary frame
                    # # Use dilation to extend the boundary pixels to the edges between neighboring pixels
                    # boundary_mask = dilation(boundary_mask)

                    # Extract coordinates of the boundary pixels
                    # boundary_coords = np.argwhere(boundary_mask)
                    # new_mask =outside_boundary_mask
                    # boundary_coords = np.argwhere(new_mask)
                    coords = region.coords
                    coords_2d = [(coord[0], coord[1]) for coord in coords]
                    boundary = get_polygon_outline(coords_2d)
                    # boundary_sorted = sort_points_clockwise(boundary)
                    boundary_sorted = boundary
                    # Shift the coordinates to align with the pixel edges (half-pixel shift)
                    boundary_coords_3d = [[z_index, coord[0] , coord[1]] for coord in boundary_sorted]
   

                    stacked_labels[z_index][labels_plane == region.label] = 1
                    shapes_layer.add(boundary_coords_3d, shape_type="polygon", edge_width=0.1)
                    
        layer_name = f'Thresh={num_stddevs}_MinSize={min_puncta_size}'
        viewer.add_labels(stacked_labels, name=layer_name)

# Add the normalized 4-channel image for reference
viewer.add_image(image_with_normch4, name='Image with NormCh4')
napari.run()
