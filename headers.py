"""headers.py"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model # type: ignore

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from dataset import (load_all_images, make_folds, check_sizes, make_patches,get_repeats, build_train_ds_from_arrays,)
from utils.ablation_losses import bce_dice_loss
from utils.ablation_metrics import (METRICS, confusion_counts, metrics_from_counts, summarize, stats_on_dice,)
from utils.ablation_plots import plot_violins, plot_prediction_grids
from utils.ablation_io import (make_run_dir, start_log, append_rows, save_json, save_fold_predictions,)

ALL_MODELS = ["fcn", "encdec", "unet", "resunet", "resse", "attunet", "attaspp"]

ABLATION_VARIANTS = {
    "encdec":  dict(label="Enc-Dec",       display="Encoder-decoder (no skips)"),
    "unet":    dict(label="U-Net",         display="+ skip connections"),
    "resunet": dict(label="ResUNet",       display="+ residual blocks"),
    "resse":   dict(label="ResUNet+SE",    display="+ SE in residual blocks"),
    "attunet": dict(label="Att-UNet",      display="+ attention gates"),
    "attaspp": dict(label="Att-UNet+ASPP", display="+ ASPP bridge"),
}