    def label_points(event):
        current_num_points = len(points_layer.data)
        num_existing_labels = len(points_layer.features['label'])

        if current_num_points > num_existing_labels:
            new_labels = np.arange(num_existing_labels + 1, current_num_points + 1)
            points_layer.features['label'] = np.append(points_layer.features['label'], new_labels)

            # Refresh after updating the labels
            points_layer.refresh()

        print(f"Updated Labels: {points_layer.features['label']}")
    return label_points
