"""
dataset.py
----------
Handles:
    1. Loading raw .tif fluorescence microscopy images.
    2. Classical labeling pipeline (v4 - Frangi ridge detection):
         Gaussian smoothing
         -> Frangi vesselness filter  (detects bright curvilinear actin)
         -> Percentile threshold      (binarize Frangi response)
         -> Small-component removal   (suppress speckle noise)
         -> HOG-guided refinement     (remove non-edge noise pixels)
    3. Creating tf.data.Dataset pipelines for training.

Why Frangi instead of Otsu + erosion:
    The previous Otsu-based pipeline assumed one isolated cell with actin
    only on its outer boundary. Real images are confluent monolayers where
    actin accumulates at cell-cell junctions throughout the entire field.
    The Frangi vesselness filter directly detects bright curvilinear
    ridge-like structures (actin filaments / junctions) without assuming
    anything about cell shape, cell count, or background separation.
"""

import os
import glob

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import remove_small_objects

import config
from utils.hog_processing import extract_hog_features


# ============================================================
# 1. LOAD IMAGE
# ============================================================

def load_tif_image(path: str) -> np.ndarray:
    """
    Load a TIFF fluorescence image and normalize it to [0, 1].

    Uses PIL so that 16-bit TIFFs are read correctly (OpenCV
    imread with IMREAD_UNCHANGED also works but PIL handles
    multi-page TIFFs and unusual colour modes more gracefully).

    Returns
    -------
    np.ndarray
        2D float32 array, values in [0, 1].
    """
    with Image.open(path) as img:
        arr = np.array(img, dtype=np.float32)

    # Collapse to single channel if multi-channel / RGB
    if arr.ndim == 3:
        arr = arr[..., 0]

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.float32)


# ============================================================
# 2. FRANGI-BASED LABELING PIPELINE
# ============================================================

def preprocess_image(img: np.ndarray) -> dict:
    """
    Generate a binary actin-edge label from one normalized image
    using a Frangi vesselness filter instead of the Otsu+erosion
    approach that assumed a single isolated cell.

    Steps
    -----
    1. Gaussian smoothing  -- suppress high-frequency shot noise
       so the Hessian matrix used by the Frangi filter sees clean
       intensity ridges rather than noise spikes.
    2. Frangi vesselness filter  -- computes the Hessian eigenvalues
       at multiple scales (config.FRANGI_SIGMAS). At ridge-like
       structures (actin junctions), one eigenvalue is large and
       negative while the other is near zero; the vesselness score
       combines these into a strong, scale-normalised response at
       the filaments and near-zero everywhere else.
    3. Percentile threshold  -- binarize the Frangi response map.
       Only the top (100 - FRANGI_THRESHOLD_PERCENTILE)% of pixels
       pass; because the Frangi response is extremely right-skewed,
       this naturally selects genuine actin ridges.
    4. Small-component removal  -- drop binary blobs smaller than
       FRANGI_MIN_COMPONENT_AREA pixels to suppress isolated speckle
       noise that still passed the threshold.
    5. HOG-guided refinement  (if config.USE_HOG_REFINEMENT)  --
       re-weight each candidate actin pixel by the HOG gradient-
       orientation response at that location. HOG responds strongly
       at oriented intensity edges (true membrane ridges) and weakly
       at isotropic bright spots (out-of-focus fluorescence, dust).
       Pixels that are bright in Frangi but lack directional gradient
       structure are suppressed.

    Parameters
    ----------
    img : np.ndarray
        Normalized float32 image, values in [0, 1], shape (H, W).

    Returns
    -------
    dict with keys:
        smoothed          -- Gaussian-denoised image
        frangi_map        -- raw float32 Frangi vesselness response
        frangi_binary     -- binarized Frangi map (before HOG)
        hog_weight_map    -- HOG soft-weight map (or copy of frangi_binary)
        edge_mask         -- final binary actin label used for training
        footprint         -- dilated actin mask used as "cell region"
                             proxy for biophysical metrics
    """

    # ----------------------------------------------------------
    # Step 1: Gaussian smoothing
    # ----------------------------------------------------------
    smoothed = cv2.GaussianBlur(
        img,
        config.GAUSSIAN_KERNEL_SIZE,
        sigmaX=config.GAUSSIAN_SIGMA,
    )

    # ----------------------------------------------------------
    # Step 2: Frangi vesselness filter
    #
    # skimage.filters.frangi expects float image in [0, 1].
    # black_ridges=False -> detect BRIGHT ridges on dark bg.
    # Returns a float map in [0, 1]: high where the image looks
    # like a bright curvilinear tube/sheet, near 0 elsewhere.
    # ----------------------------------------------------------
    frangi_map = frangi(
        smoothed,
        sigmas=config.FRANGI_SIGMAS,
        alpha=config.FRANGI_ALPHA,
        beta=config.FRANGI_BETA,
        black_ridges=config.FRANGI_BLACK_RIDGES,
    ).astype(np.float32)

    # ----------------------------------------------------------
    # Step 3: Percentile threshold
    # ----------------------------------------------------------
    nonzero_vals = frangi_map[frangi_map > 0]
    if nonzero_vals.size > 0:
        threshold = np.percentile(
            nonzero_vals, config.FRANGI_THRESHOLD_PERCENTILE
        )
    else:
        threshold = 1.0  # nothing detected -> empty mask

    frangi_binary = (frangi_map >= threshold).astype(np.uint8)

    # ----------------------------------------------------------
    # Step 4: Remove small components (noise suppression)
    # ----------------------------------------------------------
    if config.FRANGI_MIN_COMPONENT_AREA > 0 and frangi_binary.sum() > 0:
        cleaned = remove_small_objects(
            frangi_binary.astype(bool),
            max_size=config.FRANGI_MIN_COMPONENT_AREA,
        )
        frangi_binary = cleaned.astype(np.uint8)

    # ----------------------------------------------------------
    # Step 5: HOG-guided refinement
    # ----------------------------------------------------------
    if config.USE_HOG_REFINEMENT and frangi_binary.sum() > 0:
        _, hog_vis = extract_hog_features(smoothed, visualize=True)

        if hog_vis.shape != smoothed.shape:
            hog_vis = cv2.resize(
                hog_vis, (smoothed.shape[1], smoothed.shape[0])
            )

        # Soft-weight: blend HOG orientation response with the
        # raw binary mask so gradient-consistent pixels score
        # higher than isotropic bright spots.
        w = config.HOG_REFINEMENT_WEIGHT
        hog_weight_map = (
            frangi_binary.astype(np.float32)
            * ((1 - w) + w * hog_vis.astype(np.float32))
        ).astype(np.float32)
        hog_weight_map = np.clip(hog_weight_map, 0, 1)

        edge_mask = (
            (hog_weight_map >= config.HOG_REFINEMENT_THRESHOLD)
            & (frangi_binary > 0)
        ).astype(np.uint8)
    else:
        hog_weight_map = frangi_binary.astype(np.float32)
        edge_mask = frangi_binary.copy()

    # ----------------------------------------------------------
    # Footprint: dilate actin mask to approximate the "cell
    # region" for biophysical metrics (PER, MCC, RAT, SPI).
    # Since there is no separate cell-body segmentation in the
    # Frangi pipeline, we use a thick dilation of the actin mask
    # as a stand-in for "region near the membrane."
    # ----------------------------------------------------------
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.BAND_WIDTH_PX * 4 + 1, config.BAND_WIDTH_PX * 4 + 1),
    )
    footprint = cv2.dilate(
        edge_mask, dil_kernel, iterations=1
    ).astype(np.uint8)

    return {
        "smoothed":      smoothed,
        "frangi_map":    frangi_map,
        "frangi_binary": frangi_binary,
        "hog_weight_map": hog_weight_map,
        "edge_mask":     edge_mask,
        "footprint":     footprint,
    }


# ============================================================
# 3. NUMPY LOADER (called inside tf.py_function)
# ============================================================

def _load_and_label_numpy(path_bytes):
    """
    Load one .tif image, run the Frangi labeling pipeline,
    return (image, mask) as (H,W,1) float32 arrays.
    """
    path = path_bytes.numpy().decode("utf-8")
    img = load_tif_image(path)

    img_resized = cv2.resize(
        img, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
    )

    processed = preprocess_image(img_resized)
    mask = processed["edge_mask"].astype(np.float32)

    image_arr = img_resized[..., np.newaxis].astype(np.float32)  # (H,W,1)
    mask_arr  = mask[..., np.newaxis].astype(np.float32)          # (H,W,1)
    return image_arr, mask_arr


# ============================================================
# 4. TF.DATA WRAPPERS
# ============================================================

def _tf_load_and_label(path_tensor):
    image, mask = tf.py_function(
        func=_load_and_label_numpy,
        inp=[path_tensor],
        Tout=(tf.float32, tf.float32),
    )
    h, w = config.IMAGE_SIZE
    image.set_shape((h, w, 1))
    mask.set_shape((h, w, 1))
    return image, mask


def _load_image_only_numpy(path_bytes):
    path = path_bytes.numpy().decode("utf-8")
    img = load_tif_image(path)
    img_resized = cv2.resize(
        img, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
    )
    return img_resized[..., np.newaxis].astype(np.float32)


def _tf_load_image_only(path_tensor):
    image = tf.py_function(
        func=_load_image_only_numpy,
        inp=[path_tensor],
        Tout=tf.float32,
    )
    h, w = config.IMAGE_SIZE
    image.set_shape((h, w, 1))
    return image


# ============================================================
# 5. HELPERS
# ============================================================

def get_image_paths(image_dir=None):
    image_dir = config.RAW_IMAGE_DIR if image_dir is None else image_dir
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.tif"))
        + glob.glob(os.path.join(image_dir, "*.tiff"))
    )
    if len(paths) == 0:
        raise RuntimeError(
            f"No .tif/.tiff images found in {image_dir}. "
            f"Place your microscopy images there before training."
        )
    return paths


# ============================================================
# 6. TRAIN / VALIDATION DATASETS
# ============================================================

def get_datasets(batch_size=None, val_split=None, seed=None):
    """
    Returns (train_ds, val_ds, n_train, n_val).
    Each dataset yields (image, mask) batches, float32, (B,H,W,1).
    """
    batch_size = config.BATCH_SIZE   if batch_size is None else batch_size
    val_split  = config.VAL_SPLIT    if val_split  is None else val_split
    seed       = config.RANDOM_SEED  if seed       is None else seed

    paths   = get_image_paths()
    n_total = len(paths)
    if n_total < 2:
        raise RuntimeError("At least 2 images required for train/val split.")

    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)

    train_paths = shuffled[:n_train]
    val_paths   = shuffled[n_train:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_paths)
        .shuffle(max(len(train_paths), 1), seed=seed,
                 reshuffle_each_iteration=True)
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_paths)
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, n_train, n_val
