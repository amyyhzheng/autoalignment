import numpy as np
from skimage.io import imread
import napari
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton

# Path to your 3-channel TIFF images
image_paths = [
   '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/Automated_Puncta_Detection/Image1/With_normch4-SOM022_Image 1_MotionCorrected.tif', 
   '/Volumes/nedividata/Amy/files_for_amy_fromJoe/example_analysis_SOM022/Automated_Puncta_Detection/Image2/With_normch4-SOM022_Image 2_MotionCorrected.tif'
]

# Load images from the specified paths
images = [imread(path) for path in image_paths]

# Initial points and features for all viewers
initial_points = np.array([[0, 0, 0]])
initial_features = {
    'label': [1],
    'confidence': [1],
    'type': ['-1: Nothing']
}

# Feature color and type configurations
face_color_cycle = ['blue', 'green', 'yellow', 'red', 'magenta', 'purple']
type_choices = ['-1: Nothing', '0: SpinePSD95-NoSynTD', '1: SpinePSD95', '3: NoSynNudeSpine', '4: SynNudeSpine']

# Dictionary to hold viewers and points layers
viewers = {}
points_layers = {}

class UpdatePointTypeWidget(QWidget):
    def __init__(self, points_layer):
        super().__init__()
        self.points_layer = points_layer
        layout = QVBoxLayout()
        self.label = QLabel("Select Point Type:")
        layout.addWidget(self.label)
        
        self.combo_box = QComboBox()
        self.combo_box.addItems(type_choices)
        self.combo_box.currentTextChanged.connect(self.update_point_type)
        layout.addWidget(self.combo_box)
        
        self.setLayout(layout)

    def update_point_type(self, point_type):
        selected_points = list(self.points_layer.selected_data)
        if selected_points:
            self.points_layer.features['type'][selected_points] = point_type
            self.points_layer.feature_defaults['type'] = point_type
            self.points_layer.refresh_colors(update_color_mapping=False)
        else:
            self.points_layer.feature_defaults['type'] = point_type

class ZoomLevelWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.label = QLabel(f"Zoom Level: {self.viewer.camera.zoom:.2f}")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.viewer.camera.events.zoom.connect(self.update_zoom_level)

    def update_zoom_level(self, event):
        self.label.setText(f"Zoom Level: {self.viewer.camera.zoom:.2f}")

def create_point_label_handler(points_layer):
    """Creates a handler to manage point labeling incrementally."""
    @points_layer.events.data.connect
    def label_points(event):
        num_points = len(points_layer.data)
        points_layer.feature_defaults['label'] = (points_layer.properties['label'][-1] + 1 if num_points else 1)
    return label_points

def configure_viewer(viewer, image, viewer_index):
    """Configures each viewer with images, points layers, and widgets."""
    viewer.title = f"Viewer {viewer_index + 1}"
    ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]

    viewer.add_image(ch1, name='Ch1: Cell Fill', blending='additive', colormap='red')
    viewer.add_image(ch2, name='Ch2: Gephyrin', blending='additive', colormap='cyan')
    viewer.add_image(ch3, name='Ch3: Bassoon', blending='additive', colormap='green')

    points_layer = viewer.add_points(
        initial_points,
        features=initial_features,
        size=7,
        edge_width=0.1,
        edge_color='white',
        face_color='type',
        face_color_cycle=face_color_cycle.copy(),
        text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
    )
    points_layers[f"points_layer_{viewer_index + 1}"] = points_layer

    # Configure point labeling and type update widgets
    points_layer.feature_defaults = {'label': 1, 'confidence': 1, 'type': '-1: Nothing'}
    label_handler = create_point_label_handler(points_layer)

    update_point_type_widget = UpdatePointTypeWidget(points_layer)
    viewer.window.add_dock_widget(update_point_type_widget, name="Update Point Type")

    zoom_widget = ZoomLevelWidget(viewer)
    viewer.window.add_dock_widget(zoom_widget, name="Zoom Level")

# Create viewers and configure them
for idx, img in enumerate(images):
    v = napari.Viewer()
    viewers[f"Viewer {idx + 1}"] = v
    configure_viewer(v, img, idx)

napari.run()
