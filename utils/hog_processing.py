"""
utils/hog_processing.py
------------------------
HOG (Histogram of Oriented Gradients) feature extraction used as a
refinement step in the Frangi-based labeling pipeline.

Role in the pipeline:
    After the Frangi vesselness filter produces a binary candidate mask of
    actin ridges, HOG is used to validate each candidate pixel:
    - HOG's gradient-orientation histograms respond strongly at locations
      of sharp, oriented intensity change (true membrane / junction edges).
    - Isotropic bright spots (out-of-focus fluorescence, dust, saturated
      pixels) produce a weak HOG response even if they have high Frangi
      vesselness, so HOG acts as a geometric plausibility check.
    - The HOG visualization image (per-pixel gradient-orientation magnitude,
      rescaled to [0,1]) is used as a soft weight map: candidate pixels with
      high HOG response are kept with confidence 1.0; those with low response
      are down-weighted and may fall below the binarization threshold.

Reference:
    Dalal, N. & Triggs, B. "Histograms of oriented gradients for human
    detection." CVPR, 2005.
"""

import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure

import config


def extract_hog_features(image: np.ndarray, visualize: bool = True):
    """
    Run HOG on a single-channel float32 image.

    Parameters
    ----------
    image : np.ndarray
        2D float32 image, values in [0, 1], shape (H, W).
    visualize : bool
        If True, also returns the HOG visualization image (same spatial
        resolution as input) showing per-pixel gradient-orientation
        magnitude, contrast-stretched to [0, 1].

    Returns
    -------
    features : np.ndarray
        1D HOG descriptor vector.
    hog_image_rescaled : np.ndarray or None
        2D float32 array, same shape as input, values in [0, 1].
        None if visualize=False.
    """
    image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if visualize:
        features, hog_image = hog(
            image_u8,
            orientations=config.HOG_ORIENTATIONS,
            pixels_per_cell=config.HOG_PIXELS_PER_CELL,
            cells_per_block=config.HOG_CELLS_PER_BLOCK,
            block_norm=config.HOG_BLOCK_NORM,
            visualize=True,
            feature_vector=True,
        )
        hog_image_rescaled = exposure.rescale_intensity(
            hog_image, in_range=(0, 10)
        ).astype(np.float32)
        return features, hog_image_rescaled
    else:
        features = hog(
            image_u8,
            orientations=config.HOG_ORIENTATIONS,
            pixels_per_cell=config.HOG_PIXELS_PER_CELL,
            cells_per_block=config.HOG_CELLS_PER_BLOCK,
            block_norm=config.HOG_BLOCK_NORM,
            visualize=False,
            feature_vector=True,
        )
        return features, None
