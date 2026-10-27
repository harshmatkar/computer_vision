"""
config.py
---------
Central configuration for the Actin Segmentation Pipeline.

CHANGELOG (model-training fix):
    Frangi / labeling:
        FRANGI_THRESHOLD_PERCENTILE  94   -> 90
            Gives ~3-5% foreground instead of ~1%, reducing the class-
            imbalance ratio from 99:1 to roughly 20:1, which makes both
            the loss function and augmentation more effective.
        HOG_REFINEMENT_THRESHOLD     0.35 -> 0.25
            The 0.35 cutoff was too aggressive, discarding genuine actin
            pixels that had moderate (not strong) HOG response. 0.25
            retains more real signal without letting in much extra noise.

    Loss function (utils/losses.py):
        TVERSKY_ALPHA   0.7 -> 0.3    [CRITICAL FIX]
        TVERSKY_BETA    0.3 -> 0.7    [CRITICAL FIX]
            The original config penalised FP more than FN (alpha=0.7).
            For sparse foreground (actin is 1-5% of pixels) the model
            must be penalised MORE for missing true positives (FN) than
            for including false ones (FP). Swapping to alpha=0.3 / beta=0.7
            fixes the Tversky gradient direction for Member B's model.
        COMBO_CE_BETA   0.5 -> 0.9
            Controls pos_weight in the weighted-CE component of combo loss.
            beta=0.9 → pos_weight ≈ 9. With beta=0.5 the CE was unweighted
            (50/50 FG/BG), which collapses to all-zeros on sparse foreground.

    Data augmentation:
        AUGMENT_REPEATS = 20  [NEW]
            Each training image is presented 20 times per epoch with
            independent random augmentations. Effective dataset: 5 images
            x 20 repeats x 2 geometric variants per flip = ~200 per epoch.
            Steps per epoch: 5*20/2 = 50 (was 2-3).
        AUG_BRIGHTNESS_DELTA, AUG_CONTRAST_LOWER/UPPER, AUG_NOISE_STDDEV
            Photometric augmentation parameters for _tf_augment().

    Training budget:
        NUM_EPOCHS           50  -> 100
        EARLY_STOP_PATIENCE  10  -> 20
        LEARNING_RATE     1e-3  -> 5e-4
            Lower LR + more epochs + augmentation gives the model time to
            converge rather than overshooting or stopping too early.
            ReduceLROnPlateau in train.py will halve it automatically if
            val dice plateaus.

    Diffusion model (new, models/diffusion_seg.py):
        MODEL_REGISTRY["diffusion"] and the DIFFUSION_* block below.
        Train with:  python train.py --model diffusion
        It uses config.IMAGE_SIZE, which must be divisible by 2**DIFFUSION_DEPTH.
"""

import os
import tensorflow as tf

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_IMAGE_DIR = os.path.join(DATA_DIR, "raw")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
# METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "actin_metrics.csv")

for _d in (RAW_IMAGE_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)


# IMAGE PREPROCESSING
# This tunes gaussian blurring function, in which pixel value is set to weighted average of its neibouring pixels.
# IMAGE_SIZE           = (256, 256)

IMAGE_SIZE = (512, 512)
GAUSSIAN_KERNEL_SIZE = (3, 3) # Size of window that does averaging 
GAUSSIAN_SIGMA       = 1.0 # width of bell curve in pixel larger --> Strong blur

# FRANGI VESSELNESS FILTER
FRANGI_SIGMAS      = (1, 2, 3, 4, 5)   # detection scales in pixels
FRANGI_BLACK_RIDGES = False             # detect BRIGHT ridges on dark bg
FRANGI_ALPHA       = 0.5
FRANGI_BETA        = 0.5

# Percentile threshold applied to non-zero Frangi values.
# Lowered 94 -> 90: gives ~3-5% foreground density instead of ~1%,
# reducing the class-imbalance ratio the loss function must handle.
# If masks look too noisy, raise back toward 93; if too sparse, lower to 87.
FRANGI_THRESHOLD_PERCENTILE = 87

# Remove binary blobs smaller than this area (px). 0 = disabled.
FRANGI_MIN_COMPONENT_AREA = 8

# ---------------------------------------------------------------------------
# HOG-GUIDED REFINEMENT
# ---------------------------------------------------------------------------
USE_HOG_REFINEMENT      = True
HOG_REFINEMENT_WEIGHT   = 0.6
# Lowered 0.35 -> 0.25: less aggressive filtering, retains more genuine
# actin pixels that have moderate (not strong) directional gradient.
HOG_REFINEMENT_THRESHOLD = 0.17

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM      = "L2-Hys"

# ---------------------------------------------------------------------------
# DATA AUGMENTATION
# ---------------------------------------------------------------------------
# Number of times each training image is repeated per epoch.
# The .cache() step ensures Frangi runs only once; augmentation applies
# fresh random transforms each time a cached image is drawn.
# With 5 train images: steps_per_epoch = 5 * AUGMENT_REPEATS / BATCH_SIZE
#                                       = 5 * 20 / 2 = 50 steps/epoch
AUGMENT_REPEATS = 20

# Photometric augmentation parameters (applied to image only, not mask)
AUG_BRIGHTNESS_DELTA = 0.15    # max absolute brightness shift
AUG_CONTRAST_LOWER   = 0.7     # min contrast multiplier
AUG_CONTRAST_UPPER   = 1.3     # max contrast multiplier
AUG_NOISE_STDDEV     = 0.02    # Gaussian noise standard deviation

# ---------------------------------------------------------------------------
# BIOPHYSICAL METRIC PARAMETERS
# ---------------------------------------------------------------------------
BAND_WIDTH_PX      = 6
POLARITY_NUM_SECTORS = 16

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
IN_CHANNELS         = 1
OUT_CHANNELS        = 1
BASE_FILTERS        = 16
DEPTH               = 4
USE_DEEP_SUPERVISION = True

MODEL_REGISTRY = {
    "unetpp": {
        "loss": "bce_dice",
        "checkpoint_name": "best_model_unetpp.keras",
        "metrics_csv_name": "actin_metrics_unetpp.csv",
    },
    "attention_unet": {
        "loss": "tversky",
        "checkpoint_name": "best_model_attention_unet.keras",
        "metrics_csv_name": "actin_metrics_attention_unet.csv",
    },
    "resunetpp": {
        "loss": "combo",
        "checkpoint_name": "best_model_resunetpp.keras",
        "metrics_csv_name": "actin_metrics_resunetpp.csv",
    },
    "diffusion": {
        "display_name": "Conditional Diffusion Segmenter (SegDiff-style)",
        "loss": "noise-prediction MSE",
        "checkpoint_name": "best_model_diffusion.weights.h5",   # weights only (custom model class)
        "metrics_csv_name": "actin_metrics_diffusion.csv",
        "weights_only": True,
    },
}
DEFAULT_MODEL = "unetpp"

# Focal Tversky loss (Attention UNet / "tversky")
# CRITICAL: alpha penalises FP, beta penalises FN.
# For sparse foreground: beta >> alpha (miss an actin pixel = costly).
# Original: ALPHA=0.7, BETA=0.3 -- WRONG direction, penalised FP more.
# Fixed:    ALPHA=0.3, BETA=0.7 -- correctly penalises missed foreground.
TVERSKY_ALPHA = 0.35   # was 0.7
TVERSKY_BETA  = 0.65   # was 0.3

# Combo loss (ResUNet++ / "combo")
# COMBO_ALPHA     : balance between WCE and Dice terms (0.5 = equal)
# COMBO_CE_BETA   : controls pos_weight in WCE.
#                   get_loss_fn() computes pos_weight = beta/(1-beta).
#                   beta=0.9 -> pos_weight≈9: foreground 9x more important.
#                   beta=0.5 (old) -> pos_weight=1: unweighted, collapses.
COMBO_ALPHA   = 0.5
COMBO_CE_BETA = 0.85   # was 0.5

# ---------------------------------------------------------------------------
# DIFFUSION MODEL  (models/diffusion_seg.py)
# ---------------------------------------------------------------------------
DIFFUSION_BASE_FILTERS = 16     # width of the first U-Net stage (doubles each stage)
DIFFUSION_DEPTH        = 3      # down-sampling stages; IMAGE_SIZE must be divisible by 2**DEPTH
DIFFUSION_TIME_DIM     = 64     # size of the sinusoidal time-step embedding
DIFFUSION_STEPS        = 200    # T: length of the noising chain used in training
DIFFUSION_PREDICTION   = "eps"  # "eps" = predict the added noise (standard) | "x0" = predict the mask
DIFFUSION_SAMPLING_STEPS = 20   # DDIM steps at inference (<= DIFFUSION_STEPS); fewer = faster
DIFFUSION_NUM_SAMPLES    = 4    # masks generated and averaged per image at inference
DIFFUSION_VAL_EVERY      = 5    # epochs between sampled-Dice checks on the validation set
DIFFUSION_VAL_STEPS      = 10   # sampling steps used for those checks (kept cheap)

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
BATCH_SIZE    = 2

NUM_EPOCHS    = 100

# Reduced from 1e-3 -> 5e-4: lower starting LR prevents overshooting
# with the new focal/tversky losses whose gradient scale differs from BCE.
# ReduceLROnPlateau will halve it further if val dice plateaus.
LEARNING_RATE = 5e-4

WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42

# Increased from 10 -> 20: with 50 steps/epoch and real augmentation,
# the model needs more epochs to converge. Patience of 10 fired at epoch 0
# because val dice was flat from the start (wrong loss). Now with Focal/
# Tversky losses and augmentation, val dice should improve, so we wait
# 20 epochs before early stopping.
EARLY_STOP_PATIENCE = 20

BCE_WEIGHT    = 0.5
DICE_WEIGHT   = 0.5
SEGMENTATION_PROB_THRESHOLD = 0.5
