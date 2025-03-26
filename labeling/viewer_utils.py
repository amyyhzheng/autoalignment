# import napari
# from constants import INITIAL_FEATURES, INITIAL_POINTS
# from typetocolor import TypeToColor
# from widgets import UpdatePointTypeWidget, CenterOnPointWidget, AddPointsFromCSVWidget, AddPointsLayerWidget, ZoomLevelWidget, AddPointsFromObjectJWidget
# from layer_utils import create_point_label_handler
# from constants import INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_TYPE_TO_COLOR

import napari
from constants import INITIAL_FEATURES, INITIAL_POINTS
from typetocolor import TypeToColor
from widgets import UpdatePointTypeWidget, CenterOnPointWidget, AddPointsFromCSVWidget, AddPointsLayerWidget, ZoomLevelWidget, AddPointsFromObjectJWidget
from layer_utils import create_point_label_handler
from constants import INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME, EXCITATORY_CONFOCAL_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_TYPE_TO_COLOR
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import numpy as np
from skimage.segmentation import clear_border
#TODO - Finish defining all for everything. Also maybe do default color mapping
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QCheckBox, QWidget, QLabel, QInputDialog, QApplication, QDialog
# def ask_user_for_color_mapping():
#     from PyQt5.QtWidgets import QInputDialog, QApplication
#     app = QApplication.instance()  # Get the existing QApplication
#     if not app:  # Create one if it doesn't exist
#         app = QApplication([])
    
#     #MAKE THIS MORE ADAPTABLE TO MORE COLOR MAPPINGS
#     options = [INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME]
#     mapping_name, ok = QInputDialog.getItem(
#         None, "Select Color Mapping", "Choose a color mapping:", options, 0, False
#     )
#     if ok and mapping_name:
#         return mapping_name
#     else:
#         raise ValueError("No color mapping selected. Exiting configuration.")


class MappingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Processing Options")
        self.layout = QVBoxLayout()

        self.label = QLabel("Select processing options:")
        self.layout.addWidget(self.label)

        self.map_checkbox = QCheckBox("MAP Processing")
        self.layout.addWidget(self.map_checkbox)

        self.in_vivo_checkbox = QCheckBox("In Vivo Processing")
        self.layout.addWidget(self.in_vivo_checkbox)

        options = [INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME, EXCITATORY_CONFOCAL_NAME]  
        self.mapping_name, self.ok = QInputDialog.getItem(
            self, "Select Color Mapping", "Choose a color mapping:", options, 0, False
        )

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)  # Close dialog when confirmed
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def get_results(self):
        return self.mapping_name, self.map_checkbox.isChecked(), self.in_vivo_checkbox.isChecked(), self.ok

def ask_user_mapping():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])  # Ensure an application exists

    dialog = MappingDialog()
    if dialog.exec_():  # Blocks execution until user confirms
        return dialog.get_results()
    return None, False, False, False  # Handle case when dialog is closed without confirmation



def configure_viewer(viewer, image, viewer_index):
    viewer.title = f"Viewer {viewer_index + 1}"
    mapping_name, map_check, invivo_check, ok = ask_user_mapping()
    # Ensure dialog integrates with the existing application loop
    # try:
    #     selected_mapping_name = ask_user_for_color_mapping()
    # except ValueError as e:
    #     print(f"Error during color mapping selection: {e}")
    #     return  # Exit if no mapping is selected

    if mapping_name == EXCITATORY_MAPPING_NAME:
        ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        viewer.add_image(ch1, name='Ch1: Cell Fill', blending='additive', colormap='red', scale = [3.6, 1, 1])
        viewer.layers['Ch1: Cell Fill'].contrast_limits = (3, 15)
        viewer.add_image(ch2, name='Ch2: PSD95', blending='additive', colormap='cyan', scale = [3.6, 1, 1])
        viewer.layers['Ch2: PSD95'].contrast_limits = (3, 15)
        viewer.add_image(ch3, name='Ch3: Bouton', blending='additive', colormap='green', scale = [3.6, 1, 1])
        viewer.layers['Ch3: Bouton'].contrast_limits = (3, 15)
    # Check the number of channels in the image
        
    elif mapping_name == EXCITATORY_CONFOCAL_NAME:
        z_spacing = 0.8
        ch1, ch2, ch3, ch4 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :], image[:, 3, :, :]
        viewer.add_image(ch1, name='Ch1: PSD95', blending='additive', colormap='cyan', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch2, name='Ch2: Cell Fill', blending='additive', colormap='white', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch3, name='Ch3: VGlut1', blending='additive', colormap='green', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch4, name='Ch4: Weird Channel', blending='additive', colormap='red', scale = [z_spacing, 0.12, 0.12])

    elif image.shape[1] == 4:  # 4-channel image
        #MAP Z_SPACING FOR BETTINA
        z_spacing = 0.8
        ch1, ch2, ch3, ch4 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :], image[:, 3, :, :]
        viewer.add_image(ch1, name='Ch1: RFP', blending='additive', colormap='cyan', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch2, name='Ch2: Gephyrin', blending='additive', colormap='green', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch3, name='Ch3: Cell Fill', blending='additive', colormap='white', scale = [z_spacing, 0.12, 0.12])
        viewer.add_image(ch4, name='Ch4: Bassoon', blending='additive', colormap='red', scale = [z_spacing, 0.12, 0.12])
        if map_check:
            map_processing(ch1, ch2, ch3, ch4, viewer)
    elif image.shape[1] == 3:  # 3-channel image
        ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        viewer.add_image(ch1, name='Ch1: Gephyrin', blending='additive', colormap='green', scale = [4, 1, 1])
        viewer.add_image(ch2, name='Ch2: Cell Fill', blending='additive', colormap='red', scale = [4, 1, 1])
        viewer.add_image(ch3, name='Ch3: Syntd', blending='additive', colormap='cyan', scale = [4, 1, 1]) 
        #NORMALIZED PUNCTA DETECTION - comment in/out
        if invivo_check:
            normalized_puncta(ch1, ch2, ch3, viewer)
        # viewer.layers['Ch1: RFP'] = [4, 1, 1]
        # viewer.layers['Ch2: Gephyrin'] = [4, 1, 1]
        # viewer.layers['Ch3: Cell Fill'] = [4, 1, 1]
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
        face_color= TypeToColor.map_types_to_colors(['Default'], mapping_name)[0],  # Set initial face color directly
        text={'string': '{label}', 'size': 10, 'color': 'white', 'anchor': 'center'},
        name=f"Points Layer 1",
    )
    points_layers["Points Layer 1"] = points_layer
    points_layer.feature_defaults = {
        'label': 0, 
        'Notes': 1, 
        'type': 'Default', 
    }
    # Set the face color property to use the feature
    points_layer.face_color_mode = 'direct'
    lable_handler = create_point_label_handler(points_layer, viewer)

    # Add UpdatePointTypeWidget
    update_widget = UpdatePointTypeWidget(viewer, points_layers, mapping_name)
    viewer.window.add_dock_widget(update_widget, name="Update Point Type")

    # Add AddPointsLayerWidfget
    add_layer_widget = AddPointsLayerWidget(viewer, points_layers, update_widget)
    viewer.window.add_dock_widget(add_layer_widget, name="Add Points Layer")

    center_point_widget = CenterOnPointWidget(viewer)
    viewer.window.add_dock_widget(center_point_widget, name = "Center On Point")

    # Add AddPointsFromCSVWidget
    add_csv_widget = AddPointsFromCSVWidget(viewer, points_layers, update_widget)
    viewer.window.add_dock_widget(add_csv_widget, name="Load Points from CSV")


    # Add ZoomLevelWidget
    zoom_widget = ZoomLevelWidget(viewer)
    viewer.window.add_dock_widget(zoom_widget, name="Zoom Level")

    objectj_widget = AddPointsFromObjectJWidget(viewer, points_layers, update_widget)
    viewer.window.add_dock_widget(objectj_widget, name = "Load Points from ObjectJ")
    
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

def normalized_puncta(ch1, ch2, ch3, viewer):
    # Normalize channel 2 using channel 1
    z, x, y = ch1.shape

    ch1_copy = ch1.copy()
    gephmultiplied = ch1*100
    # Testing image commented out
    # viewer.add_image(ch2multiplied)
    normch4 = gephmultiplied/ch2
    viewer.add_image(normch4, contrast_limits =(0, 80), scale = [4, 1, 1],  blending='additive' )
        
    # Define the brightness range for dendrites in channel 1
    dendritemin = 10
    dendritemax = 80

    # Create a mask for dendrites based on brightness range in ch1
    dendrite_mask = (ch2 >= dendritemin) & (ch2 <= dendritemax)
    geph_mask = ch1_copy >= 2

    # Apply the mask to normch4
    normch4_dendrites = np.where(dendrite_mask, normch4, 0)
    normch4_gephyrin = np.where(geph_mask,normch4, 0 )
    viewer.add_image(normch4_dendrites,  blending='additive', scale = [4, 1, 1])
    viewer.add_image(normch4_gephyrin,  blending='additive', scale = [4, 1, 1])
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

        
    # Loop over threshold and minimum puncta size combinations -CHANGE HERE IF NEEDED
    num_stddevs_list = [x*0.5 for x in range(0, 7)]
    for num_stddevs in num_stddevs_list:
        threshold = mean_intensity + num_stddevs * std_intensity
        for min_puncta_size in range(3, 5):
                # Initialize a 3D array for stacking filtered mask planes
            stacked_labels = np.zeros((z, x, y), dtype=int)
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
                        stacked_labels[z_index][labels_plane == region.label] = 1

            layer_name = f'Thresh={num_stddevs}_MinSize={min_puncta_size}'
            viewer.add_labels(stacked_labels, name=layer_name, scale = [4, 1, 1],  blending='additive')
