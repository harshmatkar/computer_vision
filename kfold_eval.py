"""
kfold_eval.py
-------------
TRUE Leave-One-Out Cross-Validation (LOOCV) for all three models.

For each image i (i = 1..N):
    1. Build a FRESH, untrained model.
    2. Train it on the other N-1 images only (with augmentation).
    3. Predict on image i, which the model has never seen.
    4. Compute Dice, F1, IoU, Precision, Recall vs the Frangi label.
After all folds: per-image scores and mean +/- std across folds.

Design choices (to avoid leakage)
---------------------------------
- New model every fold (clear_session + rebuild), same seed per fold.
- NO EarlyStopping / ModelCheckpoint: those would select the best epoch
  using the held-out image. Each fold trains for a fixed number of epochs.
- ReduceLROnPlateau monitors training loss, not the test image.

Usage
-----
    python kfold_eval.py --model unetpp
    python kfold_eval.py --model attention_unet --epochs 60
    python kfold_eval.py --all

Note: ground truth here is the Frangi pseudo-label, so scores measure how
well the network reproduces the classical pipeline on unseen images, not
agreement with human annotation. Cost: N trainings per model.
"""

import os
import sys
import argparse

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from dataset import (
    load_tif_image,
    preprocess_image,
    get_image_paths,
    build_train_ds,
)
from model_builder import build_and_compile, is_multi_output, num_outputs
from train import _duplicate_target_for_multi_output
from utils.metrics import compute_metrics
from utils.ploting import _save_boxplot

ALL_MODELS = ["unetpp", "attention_unet", "resunetpp"]


def predict(model, img: np.ndarray) -> np.ndarray:
    """
    Run one forward pass on a (H,W) float32 image.
    Returns a binary (H,W) uint8 mask.
    """
    tensor  = img[np.newaxis, ..., np.newaxis].astype(np.float32)
    outputs = model(tensor, training=False)
    # UNet++ returns a list (deep supervision) -- take the deepest head
    logits  = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    prob    = tf.sigmoid(logits).numpy().squeeze()
    return (prob >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)


def train_fold_model(model_name: str, train_paths: list, epochs: int):
    """Train a fresh model on train_paths for a fixed number of epochs."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    train_ds = build_train_ds(train_paths)
    if is_multi_output(model_name):
        train_ds = _duplicate_target_for_multi_output(
            train_ds, num_outputs(model_name)
        )

    model = build_and_compile(model_name)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", mode="min", factor=0.5,
            patience=7, min_lr=1e-6, verbose=0,
        ),
    ]
    model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=0)
    return model


def run_loocv(model_name: str, epochs: int = None) -> pd.DataFrame:
    """
    Run true LOOCV for one model.

    Saves:
        outputs/loocv_results_<model_name>.csv
        outputs/loocv_boxplot_<model_name>.png  (via _save_boxplot)
    """
    epochs = epochs or config.NUM_EPOCHS

    print(f"\n{'='*60}")
    print(f"LOOCV  ->  {model_name}   (epochs per fold: {epochs})")
    print(f"{'='*60}")

    image_paths = get_image_paths("data/raw")
    N = len(image_paths)
    if N < 2:
        raise RuntimeError("Need at least 2 images for LOOCV.")
    print(f"  Images  : {N}  (each held out once, trained on {N-1})\n")

    rows = []
    for i, test_path in enumerate(image_paths):
        name        = os.path.basename(test_path)
        train_paths = [p for j, p in enumerate(image_paths) if j != i]
        print(f"  Fold {i+1}/{N}: test={name}  |  training on {len(train_paths)} images ...")

        model = train_fold_model(model_name, train_paths, epochs)

        # Evaluate on the held-out image only
        raw   = load_tif_image(test_path)
        img   = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        label = preprocess_image(img)["edge_mask"]
        pred  = predict(model, img)
        m     = compute_metrics(pred, label)

        rows.append({
            "fold":     i + 1,
            "image":    name,
            "label_px": int(label.sum()),
            "pred_px":  int(pred.sum()),
            **m,
        })
        print(
            f"    Dice={m['dice']:.4f}  F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"label={int(label.sum())}px  pred={int(pred.sum())}px"
        )

        del model
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)

    print(f"\n{'-'*60}")
    print(f"  LOOCV Summary  -  {model_name}")
    print(f"{'-'*60}")
    for metric in ("dice", "f1", "iou", "precision", "recall"):
        vals = df[metric].values
        print(
            f"  {metric:<12s}:  mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )
    print(f"{'-'*60}\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, f"loocv_results_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Results saved : {csv_path}")

    _save_boxplot(df, model_name)
    return df


def save_comparison_boxplot(all_dfs: dict):
    """Side-by-side box-plot of Dice, F1, IoU for all models."""
    metrics     = ["dice", "f1", "iou"]
    metric_lbls = ["Dice", "F1", "IoU"]
    model_names = list(all_dfs.keys())
    model_lbls  = [
        config.MODEL_REGISTRY[m]["display_name"].split("(")[0].strip()
        for m in model_names
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    n_metrics = len(metrics)
    n_models  = len(model_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=True)

    for ax, metric, metric_lbl in zip(axes, metrics, metric_lbls):
        data = [all_dfs[m][metric].values for m in model_names]
        bps  = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.45,
        )
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        for i, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals, color="black", s=28, zorder=5, alpha=0.8,
            )

        ax.set_xticks(range(1, n_models + 1))
        ax.set_xticklabels(model_lbls, fontsize=9, rotation=15, ha="right")
        ax.set_title(metric_lbl, fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Score", fontsize=12)

    fig.suptitle("LOOCV Comparison - All Three Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "loocv_comparison_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved: {path}")


def print_summary_table(all_dfs: dict):
    """Print and save a mean +/- std table for all models and metrics."""
    metrics     = ["dice", "f1", "iou", "precision", "recall"]
    metric_lbls = ["Dice", "F1", "IoU", "Precision", "Recall"]

    print(f"\n{'='*84}")
    print("  LOOCV FINAL SUMMARY - mean +/- std  (N images per model)")
    print(f"{'='*84}")
    print(f"  {'Model':<28s}" + "".join(f"{l:>14s}" for l in metric_lbls))
    print(f"  {'-'*80}")

    for mname, df in all_dfs.items():
        disp = config.MODEL_REGISTRY[mname]["display_name"][:27]
        row = f"  {disp:<28s}"
        for metric in metrics:
            vals = df[metric].values
            row += f"  {vals.mean():.3f}+/-{vals.std():.3f}"
        print(row)
    print(f"{'='*84}\n")

    summary_rows = []
    for mname, df in all_dfs.items():
        r = {"model": mname}
        for metric in metrics:
            vals = df[metric].values
            r[f"{metric}_mean"] = round(float(vals.mean()), 4)
            r[f"{metric}_std"]  = round(float(vals.std()), 4)
        summary_rows.append(r)

    summary_path = os.path.join(config.OUTPUT_DIR, "loocv_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Summary CSV saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="True LOOCV evaluation.")
    parser.add_argument("--model", type=str, default="unetpp", choices=ALL_MODELS)
    parser.add_argument("--all", action="store_true",
                        help="run LOOCV for all three models")
    parser.add_argument("--epochs", type=int, default=100,
                        help="fixed epochs per fold (default: config.NUM_EPOCHS)")
    args = parser.parse_args()

    if args.all:
        all_dfs = {m: run_loocv(m, args.epochs) for m in ALL_MODELS}
        save_comparison_boxplot(all_dfs)
        print_summary_table(all_dfs)
    else:
        run_loocv(args.model, args.epochs)