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
# NOTE: no manual/Roboflow mask directory -- labels are generated entirely by
# the classical pipeline in dataset.py (Gaussian -> Otsu -> erosion -> threshold
# -> HOG refinement). See dataset.preprocess_image().

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "actin_metrics.csv")

for _d in (RAW_IMAGE_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
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
# HOG-GUIDED LABEL REFINEMENT (Step 5 of the labeling pipeline)
# ---------------------------------------------------------------------------
# After the classical Otsu/erosion/intensity-threshold chain produces a raw
# edge mask, that mask is refined using the HOG gradient-orientation response
# (see utils/hog_processing.hog_guided_boundary_refinement): pixels that are
# intensity-bright but have no real gradient structure behind them (i.e.
# likely noise) are down-weighted before the final re-binarization.
USE_HOG_REFINEMENT = True
HOG_REFINEMENT_WEIGHT = 0.75      # 0 = ignore HOG, 1 = fully weight by HOG response
                                   # (kept > 0.5 so HOG can actually veto a pixel:
                                   #  floor value for any raw-band pixel is 1-weight,
                                   #  so weight must exceed the threshold below for
                                   #  HOG to have zero effect)
HOG_REFINEMENT_THRESHOLD = 0.5    # cutoff on the HOG-weighted confidence map

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

# Registry of the three team members' models. Each entry names the model,
# its loss function, and the checkpoint/output filenames it should use, so
# train.py / evaluate.py can be pointed at any one of them via --model.
MODEL_REGISTRY = {
    "unetpp": {
        "display_name": "Lightweight UNet++ (Depthwise Separable Convs)",
        "loss": "bce_dice",
        "checkpoint_name": "best_model_unetpp.pth",
        "metrics_csv_name": "actin_metrics_unetpp.csv",
    },
    "attention_unet": {
        "display_name": "Attention U-Net",
        "loss": "tversky",
        "checkpoint_name": "best_model_attention_unet.pth",
        "metrics_csv_name": "actin_metrics_attention_unet.csv",
    },
    "resunetpp": {
        "display_name": "ResUNet++ (Residual + SE + ASPP)",
        "loss": "combo",
        "checkpoint_name": "best_model_resunetpp.pth",
        "metrics_csv_name": "actin_metrics_resunetpp.csv",
    },
}
DEFAULT_MODEL = "unetpp"

# Tversky loss hyperparameters (Member B, Attention U-Net)
TVERSKY_ALPHA = 0.7   # weight on false positives
TVERSKY_BETA = 0.3    # weight on false negatives

# Combo loss hyperparameters (Member C, ResUNet++)
COMBO_ALPHA = 0.5     # balance between weighted-CE and Dice terms
COMBO_CE_BETA = 0.5   # weight applied to the positive (foreground) class in weighted-CE

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
