"""
dataset.py
----------
Handles:
  1. Loading raw .tif fluorescence microscopy images.
  2. Preprocessing pipeline: Gaussian smoothing -> Otsu thresholding ->
     morphological erosion (boundary band) -> secondary intensity threshold
     to obtain the weak/auto-generated actin-edge label mask.
  3. Wrapping everything into a PyTorch Dataset + DataLoader.

If a matching hand-labeled mask already exists in `config.MASK_DIR`, it is
used as ground truth. Otherwise, the auto-generated boundary-band mask
(from Step 1 of the pipeline) is used as a weak label, which is common
practice in bio-image segmentation when manual annotation is scarce.
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import config
from utils.hog_processing import compute_boundary_band_mask, hog_guided_boundary_refinement


def load_tif_image(path: str) -> np.ndarray:
    """
    Loads a .tif (or any OpenCV-readable) microscopy image as a single-channel
    float32 array normalized to [0, 1].

    Parameters
    ----------
    path : str
        Full path to the image file.

    Returns
    -------
    np.ndarray
        2D float32 array, shape (H, W), values in [0, 1].
    """
    # IMREAD_UNCHANGED preserves 16-bit depth common in scientific TIFFs
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")

    # Collapse to single channel if the TIFF has multiple channels/stacks
    if img.ndim == 3:
        img = img[..., 0]

    img = img.astype(np.float32)
    # Normalize using min-max scaling (robust to 8-bit or 16-bit source depth)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


def preprocess_image(img: np.ndarray):
    """
    Runs the full classical labeling pipeline on a single normalized image.
    This IS the labeling method for this project (no manual/Roboflow
    annotation is used) -- it deterministically converts a raw intensity
    image into a binary actin-edge label mask.

    Steps
    -----
    1. Gaussian smoothing to suppress high-frequency shot noise.
    2. Otsu's thresholding on the smoothed image -> binary cell footprint
       mask ("Global Cell Segmentation").
    3. Morphological erosion of the footprint -> boundary band mask that
       hugs the cell membrane ("Boundary Band Definition").
    4. Secondary high-intensity percentile threshold applied ONLY within
       the boundary band -> raw binary edge mask of actin accumulation
       ("Targeted Intensity Thresholding" + "Binary Mask Extraction").
    5. HOG-guided refinement: the raw edge mask from step 4 is re-weighted
       by its Histogram-of-Oriented-Gradients response (utils/hog_processing.py)
       and re-binarized. HOG orientation histograms are strong exactly at
       sharp intensity transitions (i.e. true membrane edges), so this step
       suppresses edge-mask pixels that survived the intensity threshold by
       noise alone but have no real gradient structure behind them, and
       keeps pixels that are both intensity-bright AND gradient-consistent
       with a genuine boundary.

    Returns
    -------
    dict with keys:
        'smoothed'          : Gaussian-smoothed image (float32, [0,1])
        'footprint'         : binary cell mask from Otsu (uint8, {0,1})
        'boundary_band'     : binary ring mask around the membrane (uint8, {0,1})
        'edge_mask_raw'     : binary edge mask BEFORE HOG refinement (uint8, {0,1})
        'hog_weight_map'    : soft HOG confidence map used for refinement (float32, [0,1])
        'edge_mask'         : final binary actin-edge label AFTER HOG refinement (uint8, {0,1})
    """
    # --- 1. Gaussian smoothing -------------------------------------------------
    smoothed = cv2.GaussianBlur(
        img, config.GAUSSIAN_KERNEL_SIZE, sigmaX=config.GAUSSIAN_SIGMA
    )

    # --- 2. Otsu's thresholding ("Global Cell Segmentation") ----------------
    # cv2.threshold expects uint8 input for Otsu; scale [0,1] -> [0,255]
    smoothed_u8 = (smoothed * 255).astype(np.uint8)
    otsu_val, footprint = cv2.threshold(
        smoothed_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if config.OTSU_MANUAL_FLOOR > 0:
        # Optional manual floor override for very low-SNR images
        _, footprint = cv2.threshold(
            smoothed_u8, max(otsu_val, config.OTSU_MANUAL_FLOOR), 255, cv2.THRESH_BINARY
        )
    footprint = (footprint > 0).astype(np.uint8)

    # --- 3. Morphological erosion -> boundary band ("Boundary Band Definition") --
    boundary_band, eroded = compute_boundary_band_mask(
        footprint,
        structuring_element_size=config.STRUCTURING_ELEMENT_SIZE,
        iterations=config.EROSION_ITERATIONS,
    )

    # --- 4. Secondary high-intensity threshold within the band ---------------
    # ("Targeted Intensity Thresholding" + "Binary Mask Extraction")
    band_pixel_values = smoothed[boundary_band.astype(bool)]
    if band_pixel_values.size > 0:
        intensity_cutoff = np.percentile(band_pixel_values, config.EDGE_INTENSITY_PERCENTILE)
    else:
        intensity_cutoff = 1.0  # no band pixels -> empty edge mask

    edge_mask_raw = ((smoothed >= intensity_cutoff) & (boundary_band.astype(bool))).astype(np.uint8)

    # --- 5. HOG-guided refinement --------------------------------------------
    if config.USE_HOG_REFINEMENT:
        hog_weight_map = hog_guided_boundary_refinement(
            smoothed, edge_mask_raw, hog_weight=config.HOG_REFINEMENT_WEIGHT
        )
        edge_mask = (hog_weight_map >= config.HOG_REFINEMENT_THRESHOLD).astype(np.uint8)
    else:
        hog_weight_map = edge_mask_raw.astype(np.float32)
        edge_mask = edge_mask_raw

    return {
        "smoothed": smoothed,
        "footprint": footprint,
        "boundary_band": boundary_band,
        "edge_mask_raw": edge_mask_raw,
        "hog_weight_map": hog_weight_map,
        "edge_mask": edge_mask,
    }


class ActinDataset(Dataset):
    """
    PyTorch Dataset that yields (image_tensor, mask_tensor) pairs.

    Labeling method: every image is labeled automatically by the classical
    pipeline in `preprocess_image()` -- Gaussian smoothing, Otsu thresholding,
    morphological erosion (boundary band), targeted intensity thresholding,
    and HOG-guided refinement. No manual annotation tool (Roboflow or
    otherwise) is used; this dataset IS the labeling pipeline.
    """

    def __init__(self, image_dir: str = None, image_size=None):
        self.image_dir = image_dir or config.RAW_IMAGE_DIR
        self.image_size = image_size or config.IMAGE_SIZE

        self.image_paths = sorted(
            glob.glob(os.path.join(self.image_dir, "*.tif"))
            + glob.glob(os.path.join(self.image_dir, "*.tiff"))
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No .tif/.tiff images found in {self.image_dir}. "
                f"Place your microscopy images there before training."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = load_tif_image(img_path)
        img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)

        processed = preprocess_image(img_resized)
        mask = processed["edge_mask"].astype(np.float32)

        image_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()   # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()           # (1, H, W)

        return image_tensor, mask_tensor, os.path.basename(img_path)


def get_dataloaders(batch_size: int = None, val_split: float = None, seed: int = None):
    """
    Builds train/validation DataLoaders with a reproducible random split.

    Returns
    -------
    (train_loader, val_loader) : tuple of torch.utils.data.DataLoader
    """
    batch_size = batch_size or config.BATCH_SIZE
    val_split = val_split if val_split is not None else config.VAL_SPLIT
    seed = seed or config.RANDOM_SEED

    dataset = ActinDataset()
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
