"""
config.py
---------

Central configuration for the Actin Segmentation Pipeline.
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

METRICS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "actin_metrics.csv"
)

for _d in (
    RAW_IMAGE_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR
):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------------------------

IMAGE_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# FINE GAUSSIAN
# ---------------------------------------------------------------------------
# Used for the final actin intensity threshold.

GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0


# ---------------------------------------------------------------------------
# BACKGROUND / ILLUMINATION CORRECTION
# ---------------------------------------------------------------------------

BACKGROUND_CORRECTION = True

# Large enough to remove slow illumination changes,
# but not excessively large.
BACKGROUND_SIGMA = 35.0


# ---------------------------------------------------------------------------
# LOCAL CONTRAST ENHANCEMENT
# ---------------------------------------------------------------------------
# CLAHE is applied after background correction and before
# the coarse Gaussian + Otsu footprint step.
#
# This helps when cell regions and junctions have similar
# global brightness but different local contrast.

USE_CLAHE = True

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ---------------------------------------------------------------------------
# CELL FOOTPRINT
# ---------------------------------------------------------------------------
#
# IMPORTANT:
#
# The previous sigma=4.0 was too strong for this image.
# It can blur narrow dark cell-cell junctions together.
#
# We use a much smaller blur so that:
#
#     actin fibers -> somewhat smoothed
#     cell-cell gaps -> still preserved
#
# Otsu is then applied to this image.

COARSE_FOOTPRINT_SIGMA = 1.5

# (0, 0) lets OpenCV determine the kernel from sigma.
COARSE_FOOTPRINT_KERNEL_SIZE = (0, 0)


# ---------------------------------------------------------------------------
# OTSU
# ---------------------------------------------------------------------------

OTSU_MANUAL_FLOOR = 0


# ---------------------------------------------------------------------------
# FOOTPRINT MORPHOLOGY
# ---------------------------------------------------------------------------
#
# Keep this small.
#
# A large closing operation can bridge neighboring cells.

FOOTPRINT_CLOSE_KERNEL_SIZE = 3

FOOTPRINT_KEEP_LARGEST_COMPONENT_ONLY = False


# ---------------------------------------------------------------------------
# SMALL OBJECT REMOVAL
# ---------------------------------------------------------------------------
#
# Remove tiny isolated components created by noise.
#
# Set to 0 to disable.

FOOTPRINT_MIN_COMPONENT_AREA = 20


# ---------------------------------------------------------------------------
# BOUNDARY BAND
# ---------------------------------------------------------------------------
#
# The boundary is generated as:
#
#     original footprint - eroded footprint
#
# BAND_WIDTH_PX = 4 means approximately 4 pixels
# from the cell edge are considered.

BAND_WIDTH_PX = 4

EROSION_ITERATIONS = 1

STRUCTURING_ELEMENT_SIZE = (
    2 * BAND_WIDTH_PX + 1
)


# ---------------------------------------------------------------------------
# SECONDARY ACTIN INTENSITY THRESHOLD
# ---------------------------------------------------------------------------

EDGE_INTENSITY_PERCENTILE = 70


# ---------------------------------------------------------------------------
# HOG REFINEMENT
# ---------------------------------------------------------------------------

# Keep disabled until the classical boundary is correct.
USE_HOG_REFINEMENT = False

HOG_REFINEMENT_WEIGHT = 0.75

HOG_REFINEMENT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# HOG FEATURES
# ---------------------------------------------------------------------------

HOG_ORIENTATIONS = 9

HOG_PIXELS_PER_CELL = (8, 8)

HOG_CELLS_PER_BLOCK = (2, 2)

HOG_BLOCK_NORM = "L2-Hys"


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

IN_CHANNELS = 1

OUT_CHANNELS = 1

BASE_FILTERS = 16

DEPTH = 4

USE_DEEP_SUPERVISION = True


MODEL_REGISTRY = {

    "unetpp": {
        "display_name":
            "Lightweight UNet++ (Depthwise Separable Convs)",

        "loss":
            "bce_dice",

        "checkpoint_name":
            "best_model_unetpp.keras",

        "metrics_csv_name":
            "actin_metrics_unetpp.csv",
    },

    "attention_unet": {
        "display_name":
            "Attention U-Net",

        "loss":
            "tversky",

        "checkpoint_name":
            "best_model_attention_unet.keras",

        "metrics_csv_name":
            "actin_metrics_attention_unet.csv",
    },

    "resunetpp": {
        "display_name":
            "ResUNet++ (Residual + SE + ASPP)",

        "loss":
            "combo",

        "checkpoint_name":
            "best_model_resunetpp.keras",

        "metrics_csv_name":
            "actin_metrics_resunetpp.csv",
    },
}


DEFAULT_MODEL = "unetpp"


# ---------------------------------------------------------------------------
# LOSS PARAMETERS
# ---------------------------------------------------------------------------

TVERSKY_ALPHA = 0.7

TVERSKY_BETA = 0.3

COMBO_ALPHA = 0.5

COMBO_CE_BETA = 0.5


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

GPU_AVAILABLE = (
    len(tf.config.list_physical_devices("GPU")) > 0
)

BATCH_SIZE = 2

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

VAL_SPLIT = 0.2

RANDOM_SEED = 42

EARLY_STOP_PATIENCE = 10


BCE_WEIGHT = 0.5

DICE_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

SEGMENTATION_PROB_THRESHOLD = 0.5

POLARITY_NUM_SECTORS = 16