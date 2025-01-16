import numpy as np
from tifffile import imread
import napari
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QApplication, QFileDialog
import sys
import pandas as pd
from magicgui import magicgui

# Ensure a QApplication instance exists
app = QApplication.instance()  # Check if a QApplication is already running
if not app:
    app = QApplication(sys.argv)

# Path to your 3-channel TIFF images
image_paths = [
r"Z:\Amy\Imaging\2024_8_23_1317_Coronal_Section_YFPonly\TiffSaveinLab\2024-8-30_session2.lif - 1317#1_dob2-16-24_Ms-Gephyrin-488__Rb...Ch-GFP-568_Gp-Bassoon-647__63x_0.66zstepsize_1.46zoom__Cell1A4.tif"
]
type_to_color = {
    'Ignore': 'black',
    'ShaftGephBassoonNoSynTd': 'yellow',
    'ShaftGephBassoonSynTd': 'cyan',
    'SpineGephBassoonNoSynTd': 'green',
    'SpineGephBassoonSynTd': 'blue',
    'GephNoBassoon': 'red',
    'Ambiguous': 'magenta'
}

def map_types_to_colors(types):
    """Map types to colors using the type_to_color dictionary."""
    valid_colors = []
    for t in types:
        color = type_to_color.get(t, 'black')
        if color.startswith('#'):  # Convert hex to normalized RGBA
            color = tuple(int(color.lstrip('#')[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        valid_colors.append(color)

    print(valid_colors)
    return valid_colors

# Extract the types and colors as separate lists (for reference)
type_choices = list(type_to_color.keys())
face_color_cycle = list(type_to_color.values())

# Load images from the specified paths
images = [imread(path) for path in image_paths]

# Initial points and features for all viewers
initial_points = np.array([[0, 0, 0]])
initial_features = {
    'label': [1],
    'confidence': [1],
    'type': ['Ignore'], 
    'face_color': map_types_to_colors(['Ignore'])
}

# Dictionary to hold viewers and points layers
viewers = {}
points_layers = {}

def create_point_label_handler(points_layer):
    """Creates a handler to manage point labeling incrementally."""
    @points_layer.events.data.connect
    def label_points(event):
        num_points = len(points_layer.data)
        labels = points_layer.features.get('label', pd.Series(dtype=int))

        # Check if labels exist correctly
        new_label = (labels.iloc[-1] + 1) if not labels.empty else 0
        
        points_layer.feature_defaults['label'] = new_label if num_points > 1 else 0
        print(f"Updated Labels: {points_layer.features['label']}")
    return label_points


class UpdatePointTypeWidget(QWidget):
    def __init__(self, viewer, points_layers):
        super().__init__()
        self.viewer = viewer
        #Change the points_layers so that a certain 
        self.points_layers = points_layers
        self.current_points_layer = next(iter(points_layers.values()))
        self.viewer.layers.selection.events.active.connect(self.update_active_layer)

        layout = QVBoxLayout()
        self.label = QLabel("Select Point Type:")
        layout.addWidget(self.label)

        self.combo_box = QComboBox()
        self.combo_box.addItems(type_choices)
        self.combo_box.currentTextChanged.connect(self.update_point_type)
        layout.addWidget(self.combo_box)

        self.new_type_label = QLabel("Add New Type:")
        layout.addWidget(self.new_type_label)

        self.new_type = QLineEdit()
        layout.addWidget(self.new_type)

        self.new_color_label = QLabel("New Color (#RRGGBB):")
        layout.addWidget(self.new_color_label)

        self.new_color = QLineEdit()
        self.new_color.setText("#FFFFFF")
        layout.addWidget(self.new_color)

        self.add_button = QPushButton("Add Type")
        self.add_button.clicked.connect(self.add_new_type)
        layout.addWidget(self.add_button)

        self.viewer.layers.events.changed.connect(self.update_widget_for_active_layer)

        self.setLayout(layout)

    def update_active_layer(self, event):
        """Callback to update the current active points layer."""
        active_layer = self.viewer.layers.selection.active
        if isinstance(active_layer, napari.layers.Points):
            self.current_points_layer = active_layer
        else:
            self.current_points_layer = None

    def update_point_type(self, point_type):
        """Update the type for selected points in the current points layer."""
        if self.current_points_layer is not None:
            selected_points = list(self.current_points_layer.selected_data)
            if selected_points:
                # Update 'type' feature for selected points
                for idx in selected_points:
                    self.current_points_layer.features['type'][idx] = point_type
                
                # Update all face colors based on types
                all_types = self.current_points_layer.features['type']
                new_colors = map_types_to_colors(all_types)
                
                # Update the face colors for all points
                self.current_points_layer.face_color = new_colors
                self.current_points_layer.face_color_mode = 'direct'
                
                # Force a refresh
                self.current_points_layer.refresh_colors()
            else:
                # Update default for new points
                self.current_points_layer.feature_defaults['type'] = point_type
                color = map_types_to_colors([point_type])[0]
                self.current_points_layer.feature_defaults['face_color'] = color

            
    def add_new_type(self):
        new_type = self.new_type.text()
        new_color = self.new_color.text()

        if new_type and new_color.startswith('#') and len(new_color) == 7:
            if new_type not in type_choices:
                # Add to type_to_color dictionary
                type_to_color[new_type] = new_color
                # Add to type choices
                type_choices.append(new_type)
                # Add to combo box
                self.combo_box.addItem(new_type)

                if self.current_points_layer is not None:
                    # Update colors based on current types
                    all_types = self.current_points_layer.features['type']
                    new_colors = map_types_to_colors(all_types)
                    self.current_points_layer.face_color = new_colors
                    self.current_points_layer.refresh()
                
                # Clear the input fields
                self.new_type.clear()
                self.new_color.setText('#ff5733')

    def set_points_layer(self, points_layer):
        self.points_layer = points_layer


    def update_widget_for_active_layer(self, event):
        """Update the widget when the active layer changes."""
        selected_layers = list(self.viewer.layers.selection)  # Get currently selected layers
        if selected_layers:
            active_layer = selected_layers[0]  # Get the first selected layer
            if active_layer in self.points_layers.values():
                # Update current_points_layer to the selected layer
                self.current_points_layer = active_layer
                self.update_combo_box()  # Refresh the combo box for the new layer
            else:
                # If the selected layer is not a points_layer, clear everything
                self.current_points_layer = None
                self.combo_box.clear()
        else:
            # If no layer is selected, clear everything
            self.current_points_layer = None
            self.combo_box.clear()

    def update_combo_box(self):
        """Update the combo box based on the current points layer."""
        if self.current_points_layer:
            self.combo_box.clear()
            self.combo_box.addItems(type_choices)
            self.combo_box.setCurrentText(self.current_points_layer.feature_defaults.get('type', type_choices[0]))
            self.combo_box.currentTextChanged.connect(self.update_point_type)

    def update_point_type(self, point_type):
        """Update the type for selected points in the current points layer."""
        if self.current_points_layer is not None:
            selected_points = list(self.current_points_layer.selected_data)
            if selected_points:
                # Update 'type' feature for selected points
                for idx in selected_points:
                    self.current_points_layer.features['type'][idx] = point_type
                
                # Update all face colors based on types
                all_types = self.current_points_layer.features['type']
                new_colors = map_types_to_colors(all_types)
                
                # Update the face colors for all points
                self.current_points_layer.face_color = new_colors
                self.current_points_layer.face_color_mode = 'direct'
                
                # Force a refresh
                self.current_points_layer.refresh_colors()
            else:
                # Update default for new points
                self.current_points_layer.feature_defaults['type'] = point_type
                color = map_types_to_colors([point_type])[0]
                self.current_points_layer.feature_defaults['face_color'] = color

class AddPointsLayerWidget(QWidget):
    """Widget to add new points layers dynamically."""
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget

        layout = QVBoxLayout()

        self.add_button = QPushButton("Add Points Layer")
        self.add_button.clicked.connect(self.add_points_layer)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def add_points_layer(self):
        """Add a new points layer to the viewer."""
        layer_name = f"Points Layer {len(self.points_layers) + 1}"
        points_layer = self.viewer.add_points(
            initial_points,
            features=initial_features,
            size=7,
            edge_width=0.1,
            edge_color='white',
            face_color=map_types_to_colors(initial_features['type'])[0],
            text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
            name=layer_name,
        )

        self.points_layers[layer_name] = points_layer

        # Configure point defaults with correct type and initial label
        points_layer.feature_defaults = {
            'label': 0, 
            'confidence': 1, 
            'type': 'Ignore',
            'face_color': map_types_to_colors(['Ignore'])[0]
        }
        
        # Create handler for auto-incrementing labels
        create_point_label_handler(points_layer)

        # Set initial features for the layer
        points_layer.features = {
            'label': np.array([1]),
            'confidence': np.array([1]),
            'type': np.array(['Ignore']),
            'face_color': np.array([map_types_to_colors(['Ignore'])[0]])
        }

        # Update the UpdatePointTypeWidget with the new layer
        self.update_widget.points_layers = self.points_layers
        self.update_widget.update_widget_for_active_layer(None)  # Force update

        print(f"Added new points layer: {layer_name}")


@magicgui(call_button="Center on Point")
def go_to_point(viewer: napari.Viewer, point_number: int):
    """
    Center the viewer on the specified point in the currently selected points layer.

    Parameters:
        viewer (napari.Viewer): The Napari viewer instance.
        point_number (int): The label of the point to center on.
    """
    # Check for the active layer
    active_layer = viewer.layers.selection.active
    if not isinstance(active_layer, napari.layers.Points):
        print("No points layer selected or active layer is not a Points layer.")
        return

    try:
        # Retrieve the 'label' feature from the active points layer
        labels = active_layer.features['label']

        # Find the index of the specified label using a boolean condition
        point_index = labels[labels == point_number].index[0]

        # Get the coordinates of the specified point
        point_coords = active_layer.data[point_index]
        print(f"Point coordinates: {point_coords}")

        # Center the viewer's camera on the point
        viewer.camera.center = (point_coords[0], point_coords[1], point_coords[2])  # Reverse order for camera
        viewer.camera.zoom = 8  # Adjust zoom level as needed

        if len(point_coords) > 2:
            viewer.dims.set_point(2, point_coords[2])  # Adjust for Z dimension if needed
    except (IndexError, KeyError):
        print("Point number not found in labels or 'label' feature is missing.")

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

class AddPointsFromCSVWidget(QWidget):
    """Widget to load points layer data from a CSV file."""
    def __init__(self, viewer, points_layers, update_widget):
        super().__init__()
        self.viewer = viewer
        self.points_layers = points_layers
        self.update_widget = update_widget

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Points Layer from CSV")
        self.load_button.clicked.connect(self.load_points_from_csv)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_points_from_csv(self):
        """Load points layer from a CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options = options
        )
        if not file_path:
            return  # User canceled the file dialog

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the coordinates
            points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()

            # Extract the features
            features = {
                'label': df['label'].to_numpy(),
                'confidence': df['confidence'].to_numpy(),
                'type': df['type'].values if 'type' in df.columns else np.array(['Unknown'] * len(df))
            }

            # Add a new points layer
            layer_name = f"Points Layer {len(self.points_layers) + 1}"
            points_layer = self.viewer.add_points(
                points,
                features=features,
                size=7,
                edge_width=0.1,
                edge_color='white',
                face_color = map_types_to_colors(features['type']),
                text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
                name=layer_name,
            )

            # Add the new layer to points_layers dictionary
            self.points_layers[layer_name] = points_layer
            create_point_label_handler(points_layer)
            # Update the UpdatePointTypeWidget with the new layer
            self.update_widget.points_layers = self.points_layers
            self.update_widget.update_widget_for_active_layer(None)  # Force update

            print(f"Loaded points layer from CSV: {layer_name}")
        except Exception as e:
            print(f"Error loading points layer from CSV: {e}")


def configure_viewer(viewer, image, viewer_index):
    """Configures each viewer with images, points layers, and widgets."""
    viewer.title = f"Viewer {viewer_index + 1}"
    # Check the number of channels in the image
    if image.shape[1] == 4:  # 4-channel image
        ch1, ch2, ch3, ch4 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :], image[:, 3, :, :]
        viewer.add_image(ch1, name='Ch1: RFP', blending='additive', colormap='cyan')
        viewer.add_image(ch2, name='Ch2: Gephyrin', blending='additive', colormap='green')
        viewer.add_image(ch3, name='Ch3: Cell Fill', blending='additive', colormap='white')
        viewer.add_image(ch4, name='Ch4: Bassoon', blending='additive', colormap='red')
    elif image.shape[1] == 3:  # 3-channel image
        ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        viewer.add_image(ch1, name='Ch1: Gephyrin', blending='additive', colormap='green')
        viewer.add_image(ch2, name='Ch2: Cell Fill', blending='additive', colormap='red')
        viewer.add_image(ch3, name='Ch3: SynTd', blending='additive', colormap='cyan')
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[1]}")
    
    
    # Initialize points layers and UpdatePointTypeWidget
    points_layers = {}
    points_layer = viewer.add_points(
        initial_points,
        features=initial_features,
        size=7,
        edge_width=0.1,
        edge_color='white',
        face_color=map_types_to_colors(['Ignore'])[0],  # Set initial face color directly
        text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
        name=f"Points Layer 1",
    )
    points_layers["Points Layer 1"] = points_layer
    points_layer.feature_defaults = {
        'label': 0, 
        'confidence': 1, 
        'type': 'Ignore', 
        'face_color': map_types_to_colors(['Ignore'])[0]  # Set default face color directly
    }
    # Set the face color property to use the feature
    points_layer.face_color_mode = 'direct'
    lable_handler = create_point_label_handler(points_layer)

    # Add UpdatePointTypeWidget
    update_widget = UpdatePointTypeWidget(viewer, points_layers)
    viewer.window.add_dock_widget(update_widget, name="Update Point Type")

    # Add AddPointsLayerWidget
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


# Create viewers and configure them
for idx, img in enumerate(images):
    v = napari.Viewer()
    viewers[f"Viewer {idx + 1}"] = v
    configure_viewer(v, img, idx)

napari.run()

