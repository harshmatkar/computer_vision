"""
config.py
---------
Central configuration for the Actin Segmentation Pipeline.

Pipeline summary (v4 - Frangi ridge detection):
    The classical labeling pipeline uses a Frangi vesselness filter to
    directly detect bright curvilinear actin structures (junctions/filaments)
    in confluent monolayer images, instead of the Otsu+erosion+boundary-band
    approach which assumed a single isolated cell.

    Why Frangi?
    -----------
    The Frangi filter computes eigenvalues of the Hessian matrix at multiple
    scales. At ridge-like structures (bright curvilinear lines on a darker
    background -- exactly what actin junctions look like), one eigenvalue is
    large and negative, the other near zero. The vesselness score combines
    these into a strong response at actin junctions and near-zero response
    everywhere else. No assumption about cell shape, cell count, or background
    separation is needed -- it works directly on the intensity structure of
    the filaments themselves.
"""

import os
import tensorflow as tf

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGE_DIR = os.path.join(DATA_DIR, "raw")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "actin_metrics.csv")

for _d in (RAW_IMAGE_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------------------------
IMAGE_SIZE = (256, 256)
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 1.0

# ---------------------------------------------------------------------------
# FRANGI VESSELNESS FILTER PARAMETERS
# ---------------------------------------------------------------------------
# sigmas: range of scales (in pixels) at which to detect actin ridges.
#   Actin junctions in fluorescence microscopy are typically 1-5px wide
#   at 256x256 resolution. Using multiple sigmas makes detection robust
#   to varying filament widths across images.
FRANGI_SIGMAS = (1, 2, 3, 4, 5)

# black_ridges=False: we want BRIGHT ridges (actin) on a DARKER background.
#   Set True only if you invert your images before processing.
FRANGI_BLACK_RIDGES = False

# alpha, beta: Frangi filter shape parameters controlling sensitivity to
#   blob-vs-ridge and background noise respectively.
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5

# Threshold percentile applied to the Frangi response map to binarize it.
#   Higher = only the strongest actin ridges labeled (fewer, cleaner pixels).
#   Lower  = more actin pixels captured (noisier).
#   Start at 94 -- the Frangi response is heavily skewed so only the top
#   few percent of pixels are genuine ridges.
FRANGI_THRESHOLD_PERCENTILE = 94

# Optional: after binarizing, remove isolated specks smaller than this
#   area (in pixels) to suppress noise. 0 disables this step.
FRANGI_MIN_COMPONENT_AREA = 10

# ---------------------------------------------------------------------------
# HOG-GUIDED REFINEMENT (applied after Frangi binarization)
# ---------------------------------------------------------------------------
USE_HOG_REFINEMENT = True
HOG_REFINEMENT_WEIGHT = 0.6
HOG_REFINEMENT_THRESHOLD = 0.35

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# ---------------------------------------------------------------------------
# BIOPHYSICAL METRIC PARAMETERS
# ---------------------------------------------------------------------------
# For Frangi-based labeling the "footprint" concept changes: we use a
# dilated version of the actin mask itself as the "cell region" for
# biophysical metrics, since there is no separate cell-body segmentation.
BAND_WIDTH_PX = 6
POLARITY_NUM_SECTORS = 16

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE_FILTERS = 16
DEPTH = 4
USE_DEEP_SUPERVISION = True

MODEL_REGISTRY = {
    "unetpp": {
        "display_name": "Lightweight UNet++ (Depthwise Separable Convs)",
        "loss": "bce_dice",
        "checkpoint_name": "best_model_unetpp.keras",
        "metrics_csv_name": "actin_metrics_unetpp.csv",
    },
    "attention_unet": {
        "display_name": "Attention U-Net",
        "loss": "tversky",
        "checkpoint_name": "best_model_attention_unet.keras",
        "metrics_csv_name": "actin_metrics_attention_unet.csv",
    },
    "resunetpp": {
        "display_name": "ResUNet++ (Residual + SE + ASPP)",
        "loss": "combo",
        "checkpoint_name": "best_model_resunetpp.keras",
        "metrics_csv_name": "actin_metrics_resunetpp.csv",
    },
}
DEFAULT_MODEL = "unetpp"

TVERSKY_ALPHA = 0.7
TVERSKY_BETA  = 0.3
COMBO_ALPHA   = 0.5
COMBO_CE_BETA = 0.5

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42
EARLY_STOP_PATIENCE = 10
BCE_WEIGHT    = 0.5
DICE_WEIGHT   = 0.5
SEGMENTATION_PROB_THRESHOLD = 0.5
