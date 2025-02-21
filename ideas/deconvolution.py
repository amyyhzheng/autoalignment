import numpy as np
import tifffile as tiff
from skimage.restoration import richardson_lucy
import napari

def richardson_lucy_deconv(image_tiff, psf_tiff, iterations=10, output_tiff='deconvolved.tif'):
    """
    Perform Richardson-Lucy deconvolution on each slice of a 3D TIFF image.
    
    Parameters:
    image_tiff (str): Path to the image TIFF file.
    psf_tiff (str): Path to the PSF TIFF file.
    iterations (int): Number of iterations for the deconvolution process.
    output_tiff (str): Path to save the deconvolved image.
    """
    # Load the image and PSF
    image_stack = tiff.imread(image_tiff)
    psf = tiff.imread(psf_tiff)
    
    # Ensure PSF is normalized
    psf = psf / np.sum(psf)
    
    # Check dimensions
    if image_stack.ndim != 3 or psf.ndim != 3:
        raise ValueError("Both image and PSF must be 3D TIFFs.")
    
    # Deconvolve each slice independently
    deconvolved_stack = np.zeros_like(image_stack, dtype=np.float32)
    for i in range(image_stack.shape[0]):
        deconvolved_stack[i] = richardson_lucy(image_stack[i], psf, iterations=iterations)
    
    # # Save the result
    # tiff.imwrite(output_tiff, deconvolved_stack.astype(np.float32))
    # print(f"Deconvolved image saved to {output_tiff}")
    
    return deconvolved_stack

if __name__ == "main":
    deconvolved = richardson_lucy_deconv(r"Z:\Amy\FromBettina\2840_Session9_run1_unmixed.tiff", )
    viewer = napari.Viewer()
    viewer.add_image(deconvolved)
