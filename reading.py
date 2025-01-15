def create_point_label_handler(points_layer):
    """Creates a handler to manage point labeling incrementally."""
    @points_layer.events.data.connect
    def label_points(event):
        num_points = len(points_layer.data)
        if num_points > len(points_layer.features['label']):
            # Add a new label for the newly added point
            new_label = points_layer.features['label'][-1] + 1
            points_layer.features['label'] = np.append(points_layer.features['label'], new_label)
    return label_points
