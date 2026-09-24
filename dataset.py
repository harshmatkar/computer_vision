"""
dataset.py
----------

Handles:

1. Loading raw .tif fluorescence microscopy images.

2. Classical preprocessing and automatic labeling:

       fine Gaussian
       -> background correction
       -> local contrast enhancement
       -> coarse Gaussian
       -> Otsu thresholding
       -> morphological cleanup
       -> boundary band
       -> secondary intensity threshold
       -> optional HOG refinement

3. Creating TensorFlow datasets:

       image -> binary actin-edge mask
"""

import os
import glob

import numpy as np
import cv2
import tensorflow as tf

from scipy.ndimage import binary_fill_holes

from PIL import Image

import config

from utils.hog_processing import (
    compute_boundary_band_mask,
    hog_guided_boundary_refinement,
)


# ============================================================
# 1. LOAD IMAGE
# ============================================================

def load_tif_image(path: str) -> np.ndarray:
    """
    Load a TIFF fluorescence image and normalize it to [0, 1].

    Returns
    -------
    np.ndarray
        2D float32 image with values in [0, 1].
    """
    print("Using this")
    with Image.open(path) as img:

        img_arr = np.array(
            img,
            dtype=np.float32
        )

    # If TIFF has multiple channels,
    # use the first channel.
    if img_arr.ndim == 3:

        img_arr = img_arr[..., 0]

    min_val = np.min(img_arr)

    max_val = np.max(img_arr)

    if max_val > min_val:

        img_arr = (
            img_arr - min_val
        ) / (
            max_val - min_val
        )

    else:

        img_arr = np.zeros_like(img_arr)

    return img_arr.astype(np.float32)


# ============================================================
# 2. BACKGROUND CORRECTION
# ============================================================

def _correct_background(img: np.ndarray) -> np.ndarray:
    """
    Remove slow illumination variation using a
    multiplicative background model.

    corrected = image / background
    """

    background = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=config.BACKGROUND_SIGMA
    )

    background = np.clip(
        background,
        1e-6,
        None
    )

    corrected = img / background

    c_min = corrected.min()

    c_max = corrected.max()

    if c_max > c_min:

        corrected = (
            corrected - c_min
        ) / (
            c_max - c_min
        )

    else:

        corrected = np.zeros_like(corrected)

    return corrected.astype(np.float32)


# ============================================================
# 3. CLAHE LOCAL CONTRAST ENHANCEMENT
# ============================================================

def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Improve local contrast before Otsu.

    Input:
        float image [0, 1]

    Output:
        float image [0, 1]
    """

    img_u8 = np.clip(
        img * 255.0,
        0,
        255
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(img_u8)

    return (
        enhanced.astype(np.float32)
        / 255.0
    )


# ============================================================
# 4. REMOVE SMALL COMPONENTS
# ============================================================

def _remove_small_components(
    mask: np.ndarray,
    min_area: int
) -> np.ndarray:
    """
    Remove tiny isolated components from a binary mask.
    """

    if min_area <= 0:

        return mask.astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            cleaned[
                labels == label
            ] = 1

    return cleaned


# ============================================================
# 5. CELL FOOTPRINT
# ============================================================

def _build_cell_footprint(
    img: np.ndarray
) -> np.ndarray:
    """
    Build a binary mask representing the cellular footprint.

    Pipeline:

        image
          |
          v
        background correction
          |
          v
        CLAHE
          |
          v
        coarse Gaussian
          |
          v
        Otsu
          |
          v
        small morphological closing
          |
          v
        hole filling
          |
          v
        remove tiny components
    """

    # --------------------------------------------------------
    # STEP 1
    # Background correction
    # --------------------------------------------------------

    if config.BACKGROUND_CORRECTION:

        img_for_footprint = (
            _correct_background(img)
        )

    else:

        img_for_footprint = img.copy()


    # --------------------------------------------------------
    # STEP 2
    # Local contrast enhancement
    # --------------------------------------------------------

    if config.USE_CLAHE:

        img_for_footprint = (
            _apply_clahe(
                img_for_footprint
            )
        )


    # --------------------------------------------------------
    # STEP 3
    # Coarse Gaussian
    #
    # Important:
    #
    # Keep this relatively small so that
    # narrow cell-cell junctions survive.
    # --------------------------------------------------------

    coarse = cv2.GaussianBlur(
        img_for_footprint,
        config.COARSE_FOOTPRINT_KERNEL_SIZE,
        sigmaX=config.COARSE_FOOTPRINT_SIGMA
    )


    # --------------------------------------------------------
    # STEP 4
    # Convert to uint8
    # --------------------------------------------------------

    coarse_u8 = np.clip(
        coarse * 255.0,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # STEP 5
    # OTSU
    # --------------------------------------------------------

    otsu_val, footprint = cv2.threshold(
        coarse_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # Optional manual threshold floor
    # --------------------------------------------------------

    if config.OTSU_MANUAL_FLOOR > 0:

        threshold_value = max(
            otsu_val,
            config.OTSU_MANUAL_FLOOR
        )

        _, footprint = cv2.threshold(
            coarse_u8,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )


    # --------------------------------------------------------
    # Convert to {0,1}
    # --------------------------------------------------------

    footprint = (
        footprint > 0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # STEP 6
    # Small morphological closing
    #
    # This should fill tiny holes but avoid
    # joining neighboring cells.
    # --------------------------------------------------------

    kernel_size = (
        config.FOOTPRINT_CLOSE_KERNEL_SIZE
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    footprint = cv2.morphologyEx(
        footprint,
        cv2.MORPH_CLOSE,
        close_kernel
    )


    # --------------------------------------------------------
    # STEP 7
    # Fill enclosed holes
    # --------------------------------------------------------

    footprint = binary_fill_holes(
        footprint.astype(bool)
    ).astype(np.uint8)


    # --------------------------------------------------------
    # STEP 8
    # Remove tiny isolated regions
    # --------------------------------------------------------

    footprint = _remove_small_components(
        footprint,
        config.FOOTPRINT_MIN_COMPONENT_AREA
    )


    # --------------------------------------------------------
    # STEP 9
    # Optional largest-component filtering
    # --------------------------------------------------------

    if (
        config.FOOTPRINT_KEEP_LARGEST_COMPONENT_ONLY
    ):

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                footprint,
                connectivity=8
            )
        )

        if num_labels > 1:

            largest_label = (
                1
                + np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA
                    ]
                )
            )

            footprint = (
                labels == largest_label
            ).astype(np.uint8)


    return footprint


# ============================================================
# 6. PREPROCESS + AUTOMATIC LABEL GENERATION
# ============================================================

def preprocess_image(
    img: np.ndarray
):
    """
    Generate the automatic binary actin-edge label.

    Returns a dictionary containing all important
    intermediate processing stages.
    """
    print("yes tsuing this")
    # --------------------------------------------------------
    # STEP 1
    # Fine Gaussian
    #
    # Used for final intensity thresholding.
    # --------------------------------------------------------

    smoothed = cv2.GaussianBlur(
        img,
        config.GAUSSIAN_KERNEL_SIZE,
        sigmaX=config.GAUSSIAN_SIGMA
    )


    # --------------------------------------------------------
    # STEP 2
    # Background corrected image
    #
    # Exposed for debugging.
    # --------------------------------------------------------

    if config.BACKGROUND_CORRECTION:

        background_corrected = (
            _correct_background(img)
        )

    else:

        background_corrected = img.copy()


    # --------------------------------------------------------
    # STEP 3
    # CLAHE image
    #
    # Exposed for debugging.
    # --------------------------------------------------------

    if config.USE_CLAHE:

        contrast_image = _apply_clahe(
            background_corrected
        )

    else:

        contrast_image = (
            background_corrected.copy()
        )


    # --------------------------------------------------------
    # STEP 4
    # Coarse image
    #
    # This is the actual image sent to Otsu.
    # --------------------------------------------------------

    coarse = cv2.GaussianBlur(
        contrast_image,
        config.COARSE_FOOTPRINT_KERNEL_SIZE,
        sigmaX=config.COARSE_FOOTPRINT_SIGMA
    )


    # --------------------------------------------------------
    # STEP 5
    # Cell footprint
    # --------------------------------------------------------

    footprint = _build_cell_footprint(
        img
    )


    # --------------------------------------------------------
    # STEP 6
    # Boundary band
    # --------------------------------------------------------

    boundary_band, eroded = (
        compute_boundary_band_mask(
            footprint,
            structuring_element_size=(
                config.STRUCTURING_ELEMENT_SIZE
            ),
            iterations=(
                config.EROSION_ITERATIONS
            )
        )
    )

    boundary_band = (
        boundary_band > 0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # STEP 7
    # Intensity threshold inside boundary
    # --------------------------------------------------------

    band_pixels = smoothed[
        boundary_band.astype(bool)
    ]


    if band_pixels.size > 0:

        intensity_cutoff = np.percentile(
            band_pixels,
            config.EDGE_INTENSITY_PERCENTILE
        )

        edge_mask_raw = (
            (smoothed >= intensity_cutoff)
            &
            (boundary_band.astype(bool))
        ).astype(np.uint8)

    else:

        intensity_cutoff = 1.0

        edge_mask_raw = np.zeros_like(
            smoothed,
            dtype=np.uint8
        )


    # --------------------------------------------------------
    # STEP 8
    # HOG refinement
    # --------------------------------------------------------

    if config.USE_HOG_REFINEMENT:

        hog_weight_map = (
            hog_guided_boundary_refinement(
                smoothed,
                edge_mask_raw,
                hog_weight=(
                    config.HOG_REFINEMENT_WEIGHT
                )
            )
        )

        hog_weight_map = np.asarray(
            hog_weight_map,
            dtype=np.float32
        )

        edge_mask = (
            (hog_weight_map >=
             config.HOG_REFINEMENT_THRESHOLD)
            &
            (edge_mask_raw > 0)
        ).astype(np.uint8)

    else:

        hog_weight_map = (
            edge_mask_raw.astype(np.float32)
        )

        edge_mask = (
            edge_mask_raw.copy()
        )


    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        # Original processing
        "smoothed":
            smoothed,

        # Footprint debugging
        "background_corrected":
            background_corrected,

        "contrast_image":
            contrast_image,

        "coarse":
            coarse,

        # Cell mask
        "footprint":
            footprint,

        # Boundary
        "eroded":
            eroded,

        "boundary_band":
            boundary_band,

        # Actin mask
        "edge_mask_raw":
            edge_mask_raw,

        "hog_weight_map":
            hog_weight_map,

        "edge_mask":
            edge_mask,

        "intensity_cutoff":
            intensity_cutoff,
    }


# ============================================================
# 7. NUMPY LOADER FOR TF.DATA
# ============================================================

def _load_and_label_numpy(
    path_bytes
):
    """
    Load image and generate automatic binary mask.
    """

    path = (
        path_bytes
        .numpy()
        .decode("utf-8")
    )

    img = load_tif_image(path)

    img_resized = cv2.resize(
        img,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    processed = preprocess_image(
        img_resized
    )

    mask = (
        processed["edge_mask"]
        .astype(np.float32)
    )

    image_arr = (
        img_resized[..., np.newaxis]
        .astype(np.float32)
    )

    mask_arr = (
        mask[..., np.newaxis]
        .astype(np.float32)
    )

    return image_arr, mask_arr


# ============================================================
# 8. TENSORFLOW WRAPPER
# ============================================================

def _tf_load_and_label(
    path_tensor
):

    image, mask = tf.py_function(
        func=_load_and_label_numpy,
        inp=[path_tensor],
        Tout=(
            tf.float32,
            tf.float32
        )
    )

    h, w = config.IMAGE_SIZE

    image.set_shape(
        (h, w, 1)
    )

    mask.set_shape(
        (h, w, 1)
    )

    return image, mask


# ============================================================
# 9. GET IMAGE PATHS
# ============================================================

def get_image_paths(
    image_dir=None
):

    image_dir = (
        config.RAW_IMAGE_DIR
        if image_dir is None
        else image_dir
    )

    paths = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.tif"
            )
        )
        +
        glob.glob(
            os.path.join(
                image_dir,
                "*.tiff"
            )
        )
    )

    if len(paths) == 0:

        raise RuntimeError(
            f"No .tif/.tiff images found in "
            f"{image_dir}. "
            f"Place your microscopy images there "
            f"before training."
        )

    return paths


# ============================================================
# 10. IMAGE-ONLY LOADER
# ============================================================

def _load_image_only_numpy(
    path_bytes
):

    path = (
        path_bytes
        .numpy()
        .decode("utf-8")
    )

    img = load_tif_image(path)

    img_resized = cv2.resize(
        img,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    return (
        img_resized[..., np.newaxis]
        .astype(np.float32)
    )


def _tf_load_image_only(
    path_tensor
):

    image = tf.py_function(
        func=_load_image_only_numpy,
        inp=[path_tensor],
        Tout=tf.float32
    )

    h, w = config.IMAGE_SIZE

    image.set_shape(
        (h, w, 1)
    )

    return image


# ============================================================
# 11. CREATE TRAIN / VALIDATION DATASETS
# ============================================================

def get_datasets(
    batch_size=None,
    val_split=None,
    seed=None
):
    """
    Create training and validation datasets.
    """

    batch_size = (
        config.BATCH_SIZE
        if batch_size is None
        else batch_size
    )

    val_split = (
        config.VAL_SPLIT
        if val_split is None
        else val_split
    )

    seed = (
        config.RANDOM_SEED
        if seed is None
        else seed
    )

    image_paths = get_image_paths()

    n_total = len(image_paths)

    if n_total < 2:

        raise RuntimeError(
            "At least 2 images are required "
            "for train/validation splitting."
        )

    n_val = max(
        1,
        int(n_total * val_split)
    )

    n_train = (
        n_total - n_val
    )

    rng = np.random.default_rng(
        seed
    )

    shuffled = image_paths.copy()

    rng.shuffle(
        shuffled
    )

    train_paths = (
        shuffled[:n_train]
    )

    val_paths = (
        shuffled[n_train:]
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            train_paths
        )
        .shuffle(
            buffer_size=max(
                len(train_paths),
                1
            ),
            seed=seed,
            reshuffle_each_iteration=True
        )
        .map(
            _tf_load_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(
            batch_size
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            val_paths
        )
        .map(
            _tf_load_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(
            batch_size
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )


    return (
        train_ds,
        val_ds,
        n_train,
        n_val
    )