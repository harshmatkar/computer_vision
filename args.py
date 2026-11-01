"""
args.py - command-line options for main.py (the parsed result is `a`).

IMPORTANT: this file must NOT import from models/ (models/models.py does
`from args import a`, so importing models here is a circular import).
Model names are therefore not restricted here; main.py checks them against
ALL_MODELS and stops with a clear message if a name is unknown.
If you already have an args.py, just make sure it defines these options.
"""
import argparse

import config

ap = argparse.ArgumentParser(description="Ablation study, LOOCV by default (one model per run is fine).")
ap.add_argument("--model", nargs="+", default=["unet"],
                help="model name(s) from models/models.py, e.g. --model resse")
ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="fixed epochs per fold")
ap.add_argument("--folds", type=int, default=None,
                help="default: LOOCV (one fold per image). Give K for shuffled K-fold.")
ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
ap.add_argument("--ref", default="unet", help="reference model for the statistics (needs 2+ models in one run)")
ap.add_argument("--full-images", action="store_true", help="train on whole images, not patches")
ap.add_argument("--filters", type=int, default=None, help="base filters (default config.BASE_FILTERS)")
ap.add_argument("--match-params", action="store_true",
                help="give every model the same number of parameters as unet")
ap.add_argument("--light", action="store_true", help="shortcut for --match-params --filters 8")
ap.add_argument("--patch", type=int, default=None, help="override PATCH_SIZE (stride = patch // 2)")
ap.add_argument("--name", default=None, help="run-folder name (default: built from the options)")
a = ap.parse_args()