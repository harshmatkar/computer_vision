"""
dataset.py
----------
ALL data handling lives here (and only here):
    1. Loading raw .tif fluorescence microscopy images.
    2. Classical labeling pipeline (v4 - Frangi ridge detection):
         Gaussian smoothing
         -> Frangi vesselness filter
         -> Percentile threshold
         -> Small-component removal
         -> HOG-guided refinement
    3. tf.data pipelines for normal training (train.py), with online
       augmentation that multiplies effective dataset size by
       config.AUGMENT_REPEATS per epoch.
    4. Array helpers for cross-validation / ablation / final training:
       load + label everything once, folds, patches, array-based tf.data.

CHANGELOG:
    - Added _tf_augment() and the cache -> repeat -> shuffle -> augment pattern.
    - Added build_train_ds(train_paths) for kfold_eval.py (LOOCV).
    - Section 6: load_all_images, make_folds, check_sizes, make_patches,
      get_repeats, build_train_ds_from_arrays (moved here from
      ablation_study.py so that data handling is in one place).
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

    # Step 1: Gaussian smoothing
    smoothed = cv2.GaussianBlur(
        img, config.GAUSSIAN_KERNEL_SIZE, sigmaX=config.GAUSSIAN_SIGMA,
    )

    # Step 2: Frangi vesselness filter
    # black_ridges=False -> detect BRIGHT ridges (actin on dark bg)
    frangi_map = frangi(
        smoothed,
        sigmas=config.FRANGI_SIGMAS,
        alpha=config.FRANGI_ALPHA,
        beta=config.FRANGI_BETA,
        black_ridges=config.FRANGI_BLACK_RIDGES,
    ).astype(np.float32)

    # Step 3: Percentile threshold on non-zero Frangi values
    nonzero_vals = frangi_map[frangi_map > 0]
    if nonzero_vals.size > 0:
        threshold = np.percentile(nonzero_vals, config.FRANGI_THRESHOLD_PERCENTILE)
    else:
        threshold = 1.0

    frangi_binary = (frangi_map >= threshold).astype(np.uint8)

    # Step 4: Remove small components (noise suppression)
    if config.FRANGI_MIN_COMPONENT_AREA > 0 and frangi_binary.sum() > 0:
        cleaned = remove_small_objects(
            frangi_binary.astype(bool),
            max_size=config.FRANGI_MIN_COMPONENT_AREA,
        )
        frangi_binary = cleaned.astype(np.uint8)

    # Step 5: HOG-guided refinement
    if config.USE_HOG_REFINEMENT and frangi_binary.sum() > 0:
        _, hog_vis = extract_hog_features(smoothed, visualize=True)

        if hog_vis.shape != smoothed.shape:
            hog_vis = cv2.resize(hog_vis, (smoothed.shape[1], smoothed.shape[0]))

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

    # Footprint: dilated actin mask used as cell-region proxy
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
    Expensive (Frangi at multiple sigmas) -- results are cached in the
    dataset pipelines so this runs only ONCE per image per training run.
    """
    path = path_bytes.numpy().decode("utf-8")
    img  = load_tif_image(path)

    img_resized = cv2.resize(
        img, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
    )

    processed = preprocess_image(img_resized)
    mask      = processed["edge_mask"].astype(np.float32)

    image_arr = img_resized[..., np.newaxis].astype(np.float32)  # (H,W,1)
    mask_arr  = mask[..., np.newaxis].astype(np.float32)         # (H,W,1)
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
    both image and mask. Photometric transforms (brightness, contrast,
    noise) are applied to the image only.

    Operations applied:
        1. Random horizontal flip (p=0.5)
        2. Random vertical flip   (p=0.5)
        3. Random 90 degree rotation (k in {0,1,2,3})
        4. Random brightness jitter  (image only)
        5. Random contrast jitter    (image only)
        6. Additive Gaussian noise   (image only)
    """

    # 1. Random horizontal flip
    do_lr = tf.random.uniform(()) > 0.5
    image = tf.cond(do_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    mask  = tf.cond(do_lr, lambda: tf.image.flip_left_right(mask),  lambda: mask)

    # 2. Random vertical flip
    do_ud = tf.random.uniform(()) > 0.5
    image = tf.cond(do_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    mask  = tf.cond(do_ud, lambda: tf.image.flip_up_down(mask),  lambda: mask)

    # 3. Random 90 degree rotation
    k     = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask  = tf.image.rot90(mask,  k)

    # 4. Random brightness (image only)
    image = tf.image.random_brightness(image, max_delta=config.AUG_BRIGHTNESS_DELTA)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 5. Random contrast (image only)
    image = tf.image.random_contrast(
        image, lower=config.AUG_CONTRAST_LOWER, upper=config.AUG_CONTRAST_UPPER,
    )
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 6. Additive Gaussian noise (image only)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=config.AUG_NOISE_STDDEV)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, mask

def _tf_to_patches(image, mask):
    """Cut one (H,W,1) image and its mask into a dataset of (p,p,1) patches."""
    p = config.PATCH_SIZE
    s = config.PATCH_STRIDE

    # extract_patches works on a batch, so add a batch dimension first
    img_p = tf.image.extract_patches(
        images=image[None], sizes=[1, p, p, 1],
        strides=[1, s, s, 1], rates=[1, 1, 1, 1], padding="VALID",
    )
    msk_p = tf.image.extract_patches(
        images=mask[None], sizes=[1, p, p, 1],
        strides=[1, s, s, 1], rates=[1, 1, 1, 1], padding="VALID",
    )

    # (1, rows, cols, p*p) -> (n_patches, p, p, 1)
    img_p = tf.reshape(img_p, (-1, p, p, 1))
    msk_p = tf.reshape(msk_p, (-1, p, p, 1))
    return tf.data.Dataset.from_tensor_slices((img_p, msk_p))

def get_image_paths(image_dir=None):
    image_dir = "data/raw" if image_dir is None else image_dir
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.tif"))
        + glob.glob(os.path.join(image_dir, "*.tiff"))
    )
    return paths


# ============================================================
# 5. DATASET BUILDERS (path based: train.py, kfold_eval.py)
# ============================================================
def patches_per_image():
    """How many patches one image gives with the current settings."""
    h, w = config.IMAGE_SIZE
    p, s = config.PATCH_SIZE, config.PATCH_STRIDE
    return ((h - p) // s + 1) * ((w - p) // s + 1)


def build_train_ds(train_paths, batch_size=None, seed=None):
    batch_size = config.BATCH_SIZE  if batch_size is None else batch_size
    seed       = config.RANDOM_SEED if seed       is None else seed

    n_patches = len(train_paths) * patches_per_image()

    # each image gives several patches, so repeat less to keep the epoch length similar
    repeats = max(1, config.AUGMENT_REPEATS // patches_per_image())

    return (
        tf.data.Dataset
        .from_tensor_slices(list(train_paths))
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .flat_map(_tf_to_patches)          # full image -> patches
        .cache()
        .repeat(repeats)
        .shuffle(
            buffer_size=n_patches * repeats,
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_val_ds(val_paths, batch_size=None):
    """Validation patches, no augmentation."""
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    return (
        tf.data.Dataset
        .from_tensor_slices(list(val_paths))
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .flat_map(_tf_to_patches)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_datasets(batch_size=None, val_split=None, seed=None):
    """
    Returns (train_ds, val_ds, n_train, n_val) for normal training (train.py).
    """
    val_split = config.VAL_SPLIT   if val_split is None else val_split
    seed      = config.RANDOM_SEED if seed      is None else seed

    paths   = get_image_paths("data/raw")
    n_total = len(paths)

    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    rng      = np.random.default_rng(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)

    train_paths = shuffled[:n_train]
    val_paths   = shuffled[n_train:]

    train_ds = build_train_ds(train_paths, batch_size=batch_size, seed=seed)
    val_ds   = build_val_ds(val_paths, batch_size=batch_size)

    return train_ds, val_ds, n_train, n_val


# ============================================================
# 6. ARRAY HELPERS (cross-validation, ablation, final training)
#    Everything is loaded and labelled ONCE into numpy arrays, so
#    the slow Frangi step is not repeated per fold / per model.
# ============================================================

def _hw(v):
    """Accept an int (256) or a tuple/list ((256, 256)); return (h, w)."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def load_all_images(image_dir=None):
    """Load, resize and label EVERY image once. Returns names, X, Y (N,H,W,1)."""
    paths = get_image_paths(image_dir)
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images.")
    names, xs, ys = [], [], []
    for p in paths:
        raw = load_tif_image(p)
        img = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = preprocess_image(img)["edge_mask"]
        names.append(os.path.basename(p))
        xs.append(img[..., None].astype(np.float32))
        ys.append(mask[..., None].astype(np.float32))
    return names, np.stack(xs), np.stack(ys)


def make_folds(n: int, k, seed: int):
    """k=None (or k>=n): leave-one-out, one image per fold.
    Otherwise shuffle indices once and split into k folds that partition the data."""
    if k is None or k >= n:
        return [np.array([i]) for i in range(n)]
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def check_sizes(depth, full_images=False):
    """U-Net style nets need sizes divisible by 2**depth (patch AND full image)."""
    m = 2 ** depth
    ih, iw = _hw(config.IMAGE_SIZE)
    items = [("IMAGE height", ih), ("IMAGE width", iw)]
    if not full_images:
        ph, pw = _hw(config.PATCH_SIZE)
        items += [("PATCH height", ph), ("PATCH width", pw)]
        if ph > ih or pw > iw:
            raise ValueError("PATCH_SIZE is larger than IMAGE_SIZE")
    bad = [(nm, v) for nm, v in items if v % m != 0]
    if bad:
        raise ValueError(f"These must be divisible by 2**DEPTH = {m}: {bad}")


def make_patches(X, Y):
    """Cut image arrays into patches with config.PATCH_SIZE / PATCH_STRIDE.
    Same grid as _tf_to_patches (VALID padding)."""
    ph, pw = _hw(config.PATCH_SIZE)
    sh, sw = _hw(config.PATCH_STRIDE)
    n, h, w, _ = X.shape
    xs, ys = [], []
    for i in range(n):
        for r in range(0, h - ph + 1, sh):
            for c in range(0, w - pw + 1, sw):
                xs.append(X[i, r:r + ph, c:c + pw])
                ys.append(Y[i, r:r + ph, c:c + pw])
    return np.stack(xs), np.stack(ys)


def get_repeats(full_images=False):
    """How often the training set is repeated per epoch (same logic as build_train_ds)."""
    if full_images:
        return max(1, config.AUGMENT_REPEATS)
    return max(1, config.AUGMENT_REPEATS // patches_per_image())


def build_train_ds_from_arrays(Xp, Yp, seed=None, repeats=1, batch_size=None):
    """cache -> repeat -> shuffle -> augment -> batch, from preloaded arrays."""
    batch_size = config.BATCH_SIZE  if batch_size is None else batch_size
    seed       = config.RANDOM_SEED if seed       is None else seed
    n = len(Xp)
    return (
        tf.data.Dataset.from_tensor_slices((Xp, Yp))
        .cache()
        .repeat(repeats)
        .shuffle(n * repeats, seed=seed, reshuffle_each_iteration=True)
        .map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ============================================================
# 7. PREDICTION-TIME HELPERS (used by predict.py)
# ============================================================

def load_images_for_prediction(folder):
    """Load UNLABELLED images for inference. Returns a list of
    (name, image (H,W,1) float32 resized to config.IMAGE_SIZE, original (h, w))."""
    out = []
    for p in get_image_paths(folder):
        raw = load_tif_image(p)
        img = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        out.append((os.path.splitext(os.path.basename(p))[0],
                    img[..., None].astype(np.float32), raw.shape))
    return out


def save_mask(path, mask, original_shape):
    """Resize a binary mask back to the original image size and save it as a
    0/255 PNG."""
    h, w = original_shape
    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(mask * 255).save(path)