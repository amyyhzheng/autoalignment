import pandas as pd

def create_point_label_handler(points_layer):
    """
    Creates a handler to manage point labeling incrementally.

    :param points_layer: The points layer to attach the handler to.
    """
    @points_layer.events.data.connect
    def label_points(event):
        num_points = len(points_layer.data)
        labels = points_layer.features.get('label', pd.Series(dtype=int))

        # Check if labels exist correctly
        new_label = (labels.iloc[-1] + 1) if not labels.empty else 0

        points_layer.feature_defaults['label'] = new_label if num_points > 1 else 0
        print(f"Updated Labels: {points_layer.features['label']}")
    return label_points
