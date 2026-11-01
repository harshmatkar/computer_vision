"""
utils/ablation_io.py
--------------------
Saving of ablation results. Nothing is ever overwritten:

    <OUTPUT_DIR>/ablation/<timestamp>_<run name>/
        log.txt                    everything printed to the console
        run_config.json            command, flags, filters, parameters, folds, config values
        ablation_per_image.csv     written incrementally (after every model x fold)
        ablation_per_fold.csv, ablation_summary.csv, ablation_confusion.csv
        ablation_stats.csv, ablation_violins.png, ablation_predictions.png, ablation_errors.png
        predictions/fold_<n>.npz   predicted masks of every model for that fold
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd


def make_run_dir(base, name=None):
    """Create a NEW run folder (timestamp + name). Never reuses an existing one."""
    os.makedirs(base, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"{stamp}_{name}" if name else stamp
    path = os.path.join(base, run)
    i = 1
    while os.path.exists(path):
        path = os.path.join(base, f"{run}_{i}")
        i += 1
    os.makedirs(path)
    os.makedirs(os.path.join(path, "predictions"))
    return path


class _Tee:
    """Write everything printed to the console into a log file as well."""

    def __init__(self, path):
        self._file = open(path, "a", buffering=1, encoding="utf-8")
        self._out = sys.stdout

    def write(self, s):
        self._out.write(s)
        self._file.write(s)

    def flush(self):
        self._out.flush()
        self._file.flush()

    def __getattr__(self, name):          # isatty(), encoding, ...
        return getattr(self._out, name)


def start_log(run_dir):
    sys.stdout = _Tee(os.path.join(run_dir, "log.txt"))


def append_rows(path, rows):
    """Append rows (list of dicts) to a CSV, writing the header only once."""
    pd.DataFrame(rows).to_csv(path, mode="a", index=False,
                              header=not os.path.exists(path))


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_fold_predictions(run_dir, fold, preds):
    """preds: {key: uint8 mask}; key = '<model>__<image name>'."""
    np.savez_compressed(os.path.join(run_dir, "predictions", f"fold_{fold}.npz"), **preds)