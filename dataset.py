"""
dataset.py
----------
Handles:
    1. Loading raw .tif fluorescence microscopy images.
    2. Classical labeling pipeline (v4 - Frangi ridge detection):
         Gaussian smoothing
         -> Frangi vesselness filter
         -> Percentile threshold
         -> Small-component removal
         -> HOG-guided refinement
    3. Creating tf.data.Dataset pipelines for training, with online
       augmentation that multiplies effective dataset size by
       config.AUGMENT_REPEATS (default 20) per epoch.

CHANGELOG (augmentation fix):
    - Added _tf_augment(): applies random geometric (flip, rot90) and
      photometric (brightness, contrast, noise) transforms.  Geometric
      ops are applied identically to image AND mask; photometric ops
      to image only.
    - Modified get_datasets(): adds .cache() -> .repeat() -> .map(augment)
      pattern so Frangi labeling (expensive) runs only ONCE per image
      and augmentation generates fresh random variants every epoch.
      With 5 training images and AUGMENT_REPEATS=20, each epoch now
      has 100 training examples (50 batches of 2) instead of 2-3
      batches.  Over 100 epochs this is ~5000 gradient updates vs 33.
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
    Handles 16-bit TIFFs and multi-channel inputs (uses first channel).
    """
    with Image.open(path) as img:
        arr = np.array(img, dtype=np.float32)

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
    using a Frangi vesselness filter.

    Steps:
        1. Gaussian smoothing
        2. Frangi vesselness (multi-scale Hessian ridge detector)
        3. Percentile threshold on non-zero values
        4. Small-component removal
        5. HOG-guided refinement (if config.USE_HOG_REFINEMENT)

    Returns dict with keys: smoothed, frangi_map, frangi_binary,
                            hog_weight_map, edge_mask, footprint
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
    # black_ridges=False -> detect BRIGHT ridges (actin on dark bg)
    # ----------------------------------------------------------
    frangi_map = frangi(
        smoothed,
        sigmas=config.FRANGI_SIGMAS,
        alpha=config.FRANGI_ALPHA,
        beta=config.FRANGI_BETA,
        black_ridges=config.FRANGI_BLACK_RIDGES,
    ).astype(np.float32)

    # ----------------------------------------------------------
    # Step 3: Percentile threshold on non-zero Frangi values
    # config.FRANGI_THRESHOLD_PERCENTILE = 90 (changed from 94):
    #   the Frangi response is heavily right-skewed, so only the top
    #   few percent of non-zero pixels are genuine ridges. Lowering
    #   from 94 to 90 gives ~3-5% foreground instead of ~1%, reducing
    #   class imbalance severity for the loss function.
    # ----------------------------------------------------------
    nonzero_vals = frangi_map[frangi_map > 0]
    if nonzero_vals.size > 0:
        threshold = np.percentile(
            nonzero_vals, config.FRANGI_THRESHOLD_PERCENTILE
        )
    else:
        threshold = 1.0

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
    # Footprint: dilated actin mask used as cell-region proxy
    # for biophysical metrics.
    # ----------------------------------------------------------
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.BAND_WIDTH_PX * 4 + 1, config.BAND_WIDTH_PX * 4 + 1),
    )
    footprint = cv2.dilate(edge_mask, dil_kernel, iterations=1).astype(np.uint8)

    return {
        "smoothed":       smoothed,
        "frangi_map":     frangi_map,
        "frangi_binary":  frangi_binary,
        "hog_weight_map": hog_weight_map,
        "edge_mask":      edge_mask,
        "footprint":      footprint,
    }


# ============================================================
# 3. NUMPY LOADER (called inside tf.py_function)
# ============================================================

def _load_and_label_numpy(path_bytes) -> tuple:
    """
    Load and label one image. Returns (H,W,1) float32 image and mask.
    This is expensive (Frangi at multiple sigmas) -- results are cached
    in get_datasets() so this runs only ONCE per image per training run.
    """
    path = path_bytes.numpy().decode("utf-8")
    img  = load_tif_image(path)

    img_resized = cv2.resize(
        img, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
    )

    processed = preprocess_image(img_resized)
    mask      = processed["edge_mask"].astype(np.float32)

    image_arr = img_resized[..., np.newaxis].astype(np.float32)  # (H,W,1)
    mask_arr  = mask[..., np.newaxis].astype(np.float32)          # (H,W,1)
    return image_arr, mask_arr


# ============================================================
# 4. TF.DATA WRAPPERS
# ============================================================

def _tf_load_and_label(path_tensor):
    """TensorFlow wrapper around the numpy Frangi labeling pipeline."""
    image, mask = tf.py_function(
        func=_load_and_label_numpy,
        inp=[path_tensor],
        Tout=(tf.float32, tf.float32),
    )
    h, w = config.IMAGE_SIZE
    image.set_shape((h, w, 1))
    mask.set_shape((h, w, 1))
    return image, mask


def _tf_augment(image: tf.Tensor, mask: tf.Tensor):
    """
    Online data augmentation applied to each (image, mask) pair.

    Geometric transforms (flip, rotate) are applied identically to
    both image and mask to preserve spatial correspondence.
    Photometric transforms (brightness, contrast, noise) are applied
    to image only -- the mask is a binary label and must not change.

    With AUGMENT_REPEATS=20 this generates 20 statistically independent
    augmented variants of each image per epoch, multiplying the effective
    training set size 20x without loading any image more than once
    (the cache() step ensures the Frangi labeling only runs once).

    Operations applied:
        1. Random horizontal flip (p=0.5)
        2. Random vertical flip   (p=0.5)
        3. Random 90° rotation    (k in {0,1,2,3})
        4. Random brightness jitter  ±BRIGHTNESS_DELTA  (image only)
        5. Random contrast jitter    [CONTRAST_LOWER, CONTRAST_UPPER] (image only)
        6. Additive Gaussian noise   σ=NOISE_STDDEV     (image only)
    """

    # ── 1. Random horizontal flip ────────────────────────────────────
    do_lr = tf.random.uniform(()) > 0.5
    image = tf.cond(do_lr,
                    lambda: tf.image.flip_left_right(image),
                    lambda: image)
    mask  = tf.cond(do_lr,
                    lambda: tf.image.flip_left_right(mask),
                    lambda: mask)

    # ── 2. Random vertical flip ──────────────────────────────────────
    do_ud = tf.random.uniform(()) > 0.5
    image = tf.cond(do_ud,
                    lambda: tf.image.flip_up_down(image),
                    lambda: image)
    mask  = tf.cond(do_ud,
                    lambda: tf.image.flip_up_down(mask),
                    lambda: mask)

    # ── 3. Random 90° rotation (0 / 90 / 180 / 270°) ────────────────
    k     = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask  = tf.image.rot90(mask,  k)

    # ── 4. Random brightness  (image only) ───────────────────────────
    image = tf.image.random_brightness(image, max_delta=config.AUG_BRIGHTNESS_DELTA)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # ── 5. Random contrast  (image only) ─────────────────────────────
    image = tf.image.random_contrast(
        image,
        lower=config.AUG_CONTRAST_LOWER,
        upper=config.AUG_CONTRAST_UPPER,
    )
    image = tf.clip_by_value(image, 0.0, 1.0)

    # ── 6. Additive Gaussian noise  (image only) ─────────────────────
    noise = tf.random.normal(
        tf.shape(image), mean=0.0, stddev=config.AUG_NOISE_STDDEV
    )
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, mask


def _load_image_only_numpy(path_bytes):
    path = path_bytes.numpy().decode("utf-8")
    img  = load_tif_image(path)
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

    Training pipeline:
        paths
        -> map(_tf_load_and_label)    [expensive Frangi -- runs once]
        -> cache()                    [store (image, mask) in RAM, ~3 MB]
        -> repeat(AUGMENT_REPEATS)    [repeat cached data N times per epoch]
        -> shuffle(N * n_train)       [mix elements across repeats]
        -> map(_tf_augment)           [fresh random augmentation each pass]
        -> batch(batch_size)
        -> prefetch(AUTOTUNE)

    With n_train=5 and AUGMENT_REPEATS=20:
        - Steps per epoch: 5 * 20 / 2 = 50  (vs 2-3 before)
        - Over 100 epochs: 5,000 gradient updates (vs 33 before)
        - Effective dataset: 100 unique augmented variants per image

    Validation pipeline has NO augmentation (no .repeat(), no .map(augment))
    so val metrics reflect performance on the real images unmodified.
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

    rng      = np.random.default_rng(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)

    train_paths = shuffled[:n_train]
    val_paths   = shuffled[n_train:]

    # ── Training dataset (with augmentation) ──────────────────────────
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(train_paths)

        # Load + Frangi label (expensive) ─ runs exactly n_train times total
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Cache in RAM so Frangi never runs twice
        # For 5 images at (256,256,1) float32: ~2.5 MB -- trivial
        .cache()

        # Repeat the cache AUGMENT_REPEATS times per epoch
        # Each pass will get fresh random augmentation below
        .repeat(config.AUGMENT_REPEATS)

        # Shuffle across all repeats so batches mix different images
        .shuffle(
            buffer_size=n_train * config.AUGMENT_REPEATS,
            seed=seed,
            reshuffle_each_iteration=True,
        )

        # Apply random augmentation -- new random values each pass
        # because tf.random ops are re-sampled every time an element
        # flows through a .map() call, even for repeated elements.
        .map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)

        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Validation dataset (NO augmentation) ──────────────────────────
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices(val_paths)
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, n_train, n_val
