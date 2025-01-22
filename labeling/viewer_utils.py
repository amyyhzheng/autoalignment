import napari
from constants import INITIAL_FEATURES, INITIAL_POINTS
from typetocolor import TypeToColor
from widgets import UpdatePointTypeWidget, go_to_point, AddPointsFromCSVWidget, AddPointsLayerWidget, ZoomLevelWidget
from layer_utils import create_point_label_handler
from constants import INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_TYPE_TO_COLOR
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import numpy as np

#TODO - Finish defining all for everything. Also maybe do default color mapping

def ask_user_for_color_mapping():
    from PyQt5.QtWidgets import QInputDialog, QApplication
    app = QApplication.instance()  # Get the existing QApplication
    if not app:  # Create one if it doesn't exist
        app = QApplication([])
    
    #MAKE THIS MORE ADAPTABLE TO MORE COLOR MAPPINGS
    options = [INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME]
    mapping_name, ok = QInputDialog.getItem(
        None, "Select Color Mapping", "Choose a color mapping:", options, 0, False
    )
    if ok and mapping_name:
        return mapping_name
    else:
        raise ValueError("No color mapping selected. Exiting configuration.")

def configure_viewer(viewer, image, viewer_index):
    viewer.title = f"Viewer {viewer_index + 1}"

    # Ensure dialog integrates with the existing application loop
    try:
        selected_mapping_name = ask_user_for_color_mapping()
    except ValueError as e:
        print(f"Error during color mapping selection: {e}")
        return  # Exit if no mapping is selected

    # Check the number of channels in the image
    if image.shape[1] == 4:  # 4-channel image
        # Assuming 'image' is a 4D array (samples, channels, height, width)
        ch1, ch2, ch3, ch4 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :], image[:, 3, :, :]
        # Display images with the thresholding and colocalization results
        viewer.add_image(ch1, name='Ch1: Gephyrin', blending='additive', colormap='green')
        viewer.add_image(ch2, name='Ch2: RFP', blending='additive', colormap='cyan')
        viewer.add_image(ch3, name='Ch3: Cell Fill', blending='additive', colormap='white')
        viewer.add_image(ch4, name='Ch4: Bassoon', blending='additive', colormap='red')
        # map_processing(ch1, ch2, ch3, ch4, viewer)

    elif image.shape[1] == 3:  # 3-channel image
        ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        viewer.add_image(ch1, name='Ch1: RFP', blending='additive', colormap='cyan')
        viewer.add_image(ch2, name='Ch2: Gephyrin', blending='additive', colormap='green')
        viewer.add_image(ch3, name='Ch3: Cell Fill', blending='additive', colormap='white')
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[1]}")



    # Initialize points layers and UpdatePointTypeWidget
    points_layers = {}
    points_layer = viewer.add_points(
        INITIAL_POINTS,
        features=INITIAL_FEATURES,
        size=7,
        edge_width=0.1,
        edge_color='white',
        face_color= TypeToColor.map_types_to_colors(['Default'], selected_mapping_name)[0],  # Set initial face color directly
        text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
        name=f"Points Layer 1",
    )
    points_layers["Points Layer 1"] = points_layer
    points_layer.feature_defaults = {
        'label': 0, 
        'confidence': 1, 
        'type': 'Default', 
    }
    # Set the face color property to use the feature
    points_layer.face_color_mode = 'direct'
    lable_handler = create_point_label_handler(points_layer)

    # Add UpdatePointTypeWidget
    update_widget = UpdatePointTypeWidget(viewer, points_layers, selected_mapping_name)
    viewer.window.add_dock_widget(update_widget, name="Update Point Type")

    # Add AddPointsLayerWidfget
    add_layer_widget = AddPointsLayerWidget(viewer, points_layers, update_widget)
    viewer.window.add_dock_widget(add_layer_widget, name="Add Points Layer")

    # Add the go_to_point widget
    viewer.window.add_dock_widget(go_to_point)

    # Add AddPointsFromCSVWidget
    add_csv_widget = AddPointsFromCSVWidget(viewer, points_layers, update_widget)
    viewer.window.add_dock_widget(add_csv_widget, name="Load Points from CSV")

    # Add ZoomLevelWidget
    zoom_widget = ZoomLevelWidget(viewer)
    viewer.window.add_dock_widget(zoom_widget, name="Zoom Level")

def map_processing(ch1, ch2, ch3, ch4, viewer):

        # Thresholding Gephyrin (Ch1), Bassoon (Ch4), and Cell Fill (Ch3) channels using Otsu's method
        thresh_ch1 = threshold_otsu(ch1)
        thresh_ch4 = threshold_otsu(ch4)
        thresh_ch3 = threshold_otsu(ch3)

        # Apply thresholds
        ch1_thresholded = ch1 > thresh_ch1
        ch4_thresholded = ch4 > thresh_ch4
        ch3_thresholded = ch3 > thresh_ch3  # Cell Fill thresholded mask

        # Find colocalization (where both Gephyrin and Bassoon are above their thresholds)
        colocalization = np.logical_and(ch1_thresholded, ch4_thresholded)

        # Label the connected components in the thresholded images
        labeled_ch1 = label(ch1_thresholded)
        labeled_ch4 = label(ch4_thresholded)
        labeled_colocalization = label(colocalization)

        # Compute region properties (e.g., centroids) for each labeled region
        regions_ch1 = regionprops(labeled_ch1)
        regions_ch4 = regionprops(labeled_ch4)
        regions_colocalization = regionprops(labeled_colocalization)

        # Extract centroids from the regions
        centroids_ch1 = [region.centroid for region in regions_ch1]
        centroids_ch4 = [region.centroid for region in regions_ch4]
        centroids_colocalization = [region.centroid for region in regions_colocalization]

        filtered_areas = np.logical_and(labeled_colocalization, ch3_thresholded)
        labeled_regions = label(filtered_areas)
        filtered_regions = regionprops(labeled_regions)
        filtered_centroids = [region.centroid for region in filtered_regions]

        # Add thresholded Gephyrin and Bassoon images
        viewer.add_image(ch1_thresholded, name='Ch1: Gephyrin Thresholded', blending='additive', colormap='green')
        viewer.add_image(ch4_thresholded, name='Ch4: Bassoon Thresholded', blending='additive', colormap='red')
        viewer.add_image(ch3_thresholded, name='Ch3: Cell Fill Thresholded', blending='additive', colormap='white')

        # Add colocalization mask
        viewer.add_image(colocalization, name='Colocalization (Gephyrin & Bassoon)')

        # Add filtered centroids to the points layer (only inside the cell fill)
        viewer.add_image(filtered_areas, name='Filtered Areas', blending='additive', colormap='yellow')
        viewer.add_points(filtered_centroids, name = 'Filtered Centroids', blending = 'additive')
