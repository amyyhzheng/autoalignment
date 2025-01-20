import numpy as np
from tifffile import imread
import napari
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QApplication, QFileDialog, QMessageBox
import sys
import pandas as pd
from magicgui import magicgui

from constants import INITIAL_FEATURES, INITIAL_POINTS
from layer_utils import create_point_label_handler
from typetocolor import TypeToColor

class UpdatePointTypeWidget(QWidget):
    def __init__(self, viewer, points_layers, mapping_name):

        '''
        Currently following logic that each viewer 
        has it's own separate color mapping/types -> If 
        color mapping/types choose to be different within a single viewer, 
        Need to update the update_point_type function

        Need to add input for the color types
        '''        
        super().__init__()
        self.viewer = viewer

        self.points_layers = points_layers
        self.current_points_layer = next(iter(points_layers.values()))
        self.viewer.layers.selection.events.active.connect(self.update_active_layer)

        #Define the mapping names/get type to color
        self.mapping_name = mapping_name
        self.type_to_color = TypeToColor.get_mapping(mapping_name)
        self.type_choices = list(self.type_to_color.keys())


        layout = QVBoxLayout()
        self.label = QLabel("Select Point Type:")
        layout.addWidget(self.label)


        self.combo_box = QComboBox()
        self.combo_box.addItems(self.type_choices)
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
                new_colors = TypeToColor.map_types_to_colors(all_types, self.mapping_name)
                
                # Update the face colors for all points
                self.current_points_layer.face_color = new_colors
                self.current_points_layer.face_color_mode = 'direct'
                
                # Force a refresh
                self.current_points_layer.refresh_colors()
            else:
                # Update default for new points
                self.current_points_layer.feature_defaults['type'] = point_type
                color = TypeToColor.map_types_to_colors(point_type, self.mapping_name)[0]
                self.current_points_layer.feature_defaults['face_color'] = color

            
    def add_new_type(self):
        '''
        Updated to update the TypeToColor class 
        '''
        new_type = self.new_type.text()
        new_color = self.new_color.text()

        if not new_type:
            QMessageBox.warning(self, "Invalid Input", "Type name cannot be empty.")
            return

        if not (new_color.startswith('#') and len(new_color) == 7):
            QMessageBox.warning(self, "Invalid Input", "Color must be in the format #RRGGBB.")
            return

        # Check if the type already exists
        if new_type in self.type_choices:
            QMessageBox.warning(self, "Duplicate Type", f"Type '{new_type}' already exists.")
            return
        
        # Add the new type and color to the TypeToColor mapping
        TypeToColor.update_mapping(self.mapping_name, new_type, new_color)

        # Update the type_choices and combo box
        self.type_choices.append(new_type)
        self.combo_box.addItem(new_type)

        # Refresh the colors of the current points layer
        if self.current_points_layer is not None:
            all_types = self.current_points_layer.features['type']
            new_colors = TypeToColor.map_types_to_colors(all_types, self.mapping_name)
            self.current_points_layer.face_color = new_colors
            self.current_points_layer.face_color_mode = 'direct'
            self.current_points_layer.refresh_colors()

        # Clear the input fields
        self.new_type.clear()
        self.new_color.setText("#FFFFFF")

    def update_widget_for_active_layer(self, event):
        '''Update the widget when the active layer changes.'''
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
'''

# Can edit later - Functions to update the widget based on the active layer. 


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
                self.current_points_layer.feature_defaults['face_color'] = color'''

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
            INITIAL_POINTS,
            features=INITIAL_FEATURES,
            size=7,
            edge_width=0.1,
            edge_color='white',
            face_color= TypeToColor.map_types_to_colors(INITIAL_FEATURES['type'], self.update_widget.mapping_name)[0],
            text={'text': 'label', 'size': 10, 'color': 'white', 'anchor': 'center'},
            name=layer_name,
        )

        self.points_layers[layer_name] = points_layer

        # Configure point defaults with correct type and initial label
        points_layer.feature_defaults = {
            'label': 0, 
            'confidence': 1, 
            'type': 'Ignore'
        }
        
        # Create handler for auto-incrementing labels
        create_point_label_handler(points_layer)

        # Set initial features for the layer
        points_layer.features = {
            'label': np.array([1]),
            'confidence': np.array([1]),
            'type': np.array(['Ignore'])
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
                face_color = TypeToColor.map_types_to_colors(features['type'], self.update_widget.mapping_name),
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
