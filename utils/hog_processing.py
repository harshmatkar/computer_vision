"""
utils/hog_processing.py
------------------------
Two related responsibilities:

1. compute_boundary_band_mask(): morphological erosion utility used by
   dataset.py to build the "boundary band" ring mask that hugs the cell
   membrane (footprint - eroded_footprint = band strictly on the perimeter).

2. HOG (Histogram of Oriented Gradients) feature extraction, used here NOT
   as a general-purpose descriptor for classification, but specifically as
   a *boundary guidance signal*: HOG's gradient-orientation histograms are
   strong at locations of sharp intensity change (i.e., membrane edges),
   so we use the HOG gradient-magnitude response, re-mapped back onto the
   image grid, to reinforce/validate the auto-generated boundary band mask
   before the secondary intensity threshold is applied in dataset.py.
"""

import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure

import config


def compute_boundary_band_mask(footprint: np.ndarray, structuring_element_size: int = 3,
                                iterations: int = 1):
    """
    Erodes a binary cell footprint mask and subtracts the eroded version from
    the original to produce a thin ring ("boundary band") strictly on the
    cell perimeter.

    Parameters
    ----------
    footprint : np.ndarray
        Binary mask (uint8, values {0,1} or {0,255}) of the whole cell body,
        typically produced by Otsu's thresholding.
    structuring_element_size : int
        Size (side length) of the square structuring element used for erosion.
        Larger values erode more aggressively per iteration.
    iterations : int
        Number of erosion iterations. Combined with structuring_element_size,
        this controls how wide the resulting boundary band is
        (~ iterations * (structuring_element_size // 2) pixels wide).

    Returns
    -------
    (band, eroded) : tuple of np.ndarray
        band   : uint8 {0,1} mask of the boundary ring.
        eroded : uint8 {0,1} mask of the eroded interior (for reference/debugging).
    """
    footprint_u8 = (footprint > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (structuring_element_size, structuring_element_size)
    )
    eroded = cv2.erode(footprint_u8, kernel, iterations=iterations)
    band = footprint_u8 - eroded
    band = np.clip(band, 0, 1).astype(np.uint8)
    return band, eroded


def extract_hog_features(image: np.ndarray, visualize: bool = True):
    """
    Runs HOG on a single-channel image and returns both the raw feature
    vector (for downstream ML use, e.g. as auxiliary features) and a
    per-pixel gradient-orientation visualization image that is the same
    size as the input -- this visualization is what we use as a boundary
    guidance map.

    Parameters
    ----------
    image : np.ndarray
        2D float32 image, values in [0, 1].
    visualize : bool
        If True, also computes and returns the HOG visualization image.

    Returns
    -------
    features : np.ndarray
        1D HOG descriptor vector.
    hog_image_rescaled : np.ndarray or None
        2D float32 image (same shape as input) showing per-pixel gradient
        orientation strength, contrast-stretched to [0, 1]. None if
        visualize=False.
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
        # Contrast-stretch the visualization so weak gradients are still visible
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return features, hog_image_rescaled.astype(np.float32)
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


def hog_guided_boundary_refinement(image: np.ndarray, boundary_band: np.ndarray,
                                    hog_weight: float = 0.5) -> np.ndarray:
    """
    Refines a boundary band mask using the HOG gradient-orientation
    visualization as guidance: pixels within the original band that also
    have a strong HOG gradient response are kept with higher confidence,
    which helps suppress spurious band pixels caused by noise rather than
    true membrane edges.

    Parameters
    ----------
    image : np.ndarray
        2D float32 source image, [0, 1].
    boundary_band : np.ndarray
        uint8 {0,1} band mask from compute_boundary_band_mask().
    hog_weight : float
        Blend weight in [0, 1] controlling how strongly the HOG response
        modulates the band (0 = ignore HOG, 1 = fully weight by HOG).

    Returns
    -------
    refined_band_weight : np.ndarray
        float32 array, same shape as input, values in [0, 1] representing
        a *soft* (confidence-weighted) boundary band, where 1.0 = original
        band pixel with strong gradient support.
    """
    _, hog_vis = extract_hog_features(image, visualize=True)

    # HOG visualization can be a different internal cell resolution; resize
    # back to the exact image shape for pixel-wise multiplication.
    if hog_vis.shape != image.shape:
        hog_vis = cv2.resize(hog_vis, (image.shape[1], image.shape[0]))

    band_f = boundary_band.astype(np.float32)
    refined = band_f * ((1 - hog_weight) + hog_weight * hog_vis)
    refined = np.clip(refined, 0, 1)
    return refined
