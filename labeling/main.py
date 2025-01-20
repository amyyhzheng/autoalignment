import napari
from viewer_utils import configure_viewer
from tifffile import imread
from constants import INHIBITORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_MAPPING_NAME, EXCITATORY_TYPE_TO_COLOR
from typetocolor import TypeToColor

if __name__ == "__main__":
    image_paths = [
    '/Users/amyzheng/Desktop/4056B14ApicalsmoothedR1.tiff'
    ]
    images = [imread(path) for path in image_paths]
    TypeToColor.add_mapping(INHIBITORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR)
    TypeToColor.add_mapping(EXCITATORY_MAPPING_NAME, EXCITATORY_TYPE_TO_COLOR)

    viewers = {}
    # Create viewers and configure them
    for idx, img in enumerate(images):
        v = napari.Viewer()
        viewers[f"Viewer {idx + 1}"] = v
        configure_viewer(v, img, idx)

    napari.run()
