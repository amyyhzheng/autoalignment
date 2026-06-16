import napari
from viewer_utils import configure_viewer
from tifffile import imread
from constants import INHIBITORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR, EXCITATORY_MAPPING_NAME, EXCITATORY_TYPE_TO_COLOR, EXCITATORY_CONFOCAL_NAME, EXC_CONFOCAL_TYPE_TO_COLOR
from typetocolor import TypeToColor

if __name__ == "__main__":
    #insert image file path(s) here, separated by comma if multiple images
    image_paths = [
    r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image0.tif',
    #r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image1.tif',
    #r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image2.tif',
    #r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image3.tif',
    #r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image4.tif',
    #r'Z:\Joe\2p_data\SOM\ThirdRound\SOM056_DOB051922_RV\Analysis_withAmyCode_cell1\Images\3chImages_FullStacks\blinded\SOM056_Image5.tif',
    ]
    images = [imread(path) for path in image_paths]
    TypeToColor.add_mapping(INHIBITORY_MAPPING_NAME, INHIBITORY_TYPE_TO_COLOR)
    TypeToColor.add_mapping(EXCITATORY_MAPPING_NAME, EXCITATORY_TYPE_TO_COLOR)
    TypeToColor.add_mapping(EXCITATORY_CONFOCAL_NAME, EXC_CONFOCAL_TYPE_TO_COLOR)

    viewers = {}
    # Create viewers and configure them
    for idx, img in enumerate(images):
        v = napari.Viewer()
        viewers[f"Viewer {idx + 1}"] = v
        configure_viewer(v, img, idx)

    napari.run()
