"""
config.py
---------
Central configuration for the Actin Segmentation Pipeline.
All hyperparameters, thresholds, and file paths are defined here so that
every other module (dataset.py, train.py, evaluate.py, utils/*) can import
a single source of truth instead of hard-coding values.
"""

import os
import torch

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGE_DIR = os.path.join(DATA_DIR, "raw")        # input .tif microscopy images
MASK_DIR = os.path.join(DATA_DIR, "masks")           # ground-truth / generated masks

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "actin_metrics.csv")

for _d in (RAW_IMAGE_DIR, MASK_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# IMAGE PREPROCESSING PARAMETERS
# ---------------------------------------------------------------------------
IMAGE_SIZE = (256, 256)          # (H, W) — all images/masks are resized to this
GAUSSIAN_KERNEL_SIZE = (5, 5)    # kernel used for de-noising blur
GAUSSIAN_SIGMA = 1.0

# Otsu thresholding is parameter-free (computed automatically from histogram),
# but we keep a manual override / floor for edge cases with very low SNR.
OTSU_MANUAL_FLOOR = 0            # 0 disables manual floor, uses pure Otsu value

# Morphological erosion used to build the "boundary band" that hugs the
# cell membrane. BAND_WIDTH_PX controls how many pixels wide that band is.
BAND_WIDTH_PX = 6
EROSION_ITERATIONS = 1
STRUCTURING_ELEMENT_SIZE = 3     # size of the disk/square structuring element

# Secondary high-intensity threshold (0-255 or 0-1 depending on normalization)
# applied ONLY inside the boundary band to isolate actin accumulation.
EDGE_INTENSITY_PERCENTILE = 85   # percentile-based threshold inside the band

# ---------------------------------------------------------------------------
# HOG FEATURE EXTRACTION PARAMETERS
# ---------------------------------------------------------------------------
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# ---------------------------------------------------------------------------
# MODEL / ARCHITECTURE HYPERPARAMETERS
# ---------------------------------------------------------------------------
IN_CHANNELS = 1                  # grayscale fluorescence microscopy
OUT_CHANNELS = 1                 # binary segmentation (actin edge vs background)
BASE_FILTERS = 16                # width of first UNet++ stage (kept small -> "lightweight")
DEPTH = 4                        # number of down-sampling stages
USE_DEEP_SUPERVISION = True      # UNet++ style deep supervision on all decoder stages

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
RANDOM_SEED = 42
EARLY_STOP_PATIENCE = 10

# Loss weighting: total_loss = BCE_WEIGHT * BCEWithLogits + DICE_WEIGHT * (1 - Dice)
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# EVALUATION / BIOPHYSICAL METRIC PARAMETERS
# ---------------------------------------------------------------------------
SEGMENTATION_PROB_THRESHOLD = 0.5   # sigmoid output -> binary mask cutoff
POLARITY_NUM_SECTORS = 16           # angular sectors used for Spatial Polarity Index
