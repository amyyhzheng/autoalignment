import napari
from constants import INITIAL_FEATURES, INITIAL_POINTS
from typetocolor import TypeToColor
from widgets import UpdatePointTypeWidget, go_to_point, AddPointsFromCSVWidget, AddPointsLayerWidget, ZoomLevelWidget
from layer_utils import create_point_label_handler
from constants import INHIBITORY_MAPPING_NAME, EXCITATORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_TYPE_TO_COLOR
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
        ch1, ch2, ch3, ch4 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :], image[:, 3, :, :]
        viewer.add_image(ch1, name='Ch1: RFP', blending='additive', colormap='cyan')
        viewer.add_image(ch2, name='Ch2: Gephyrin', blending='additive', colormap='green')
        viewer.add_image(ch3, name='Ch3: Cell Fill', blending='additive', colormap='white')
        viewer.add_image(ch4, name='Ch4: Bassoon', blending='additive', colormap='red')
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
