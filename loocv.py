"""
loocv.py
--------
True Leave-One-Out Cross-Validation for all three registered models.

For each fold i (i = 1..N, N = number of images):
    - Train a FRESH model on the remaining N-1 images
    - Evaluate on image i (never seen during training for this fold)
    - Record Dice, F1, IoU, Precision, Recall

After all N folds:
    - Report mean ± std across folds  <- genuine generalisation estimate
    - Save per-model box plots
    - Save one comparison plot across all three models

Usage
-----
    python loocv.py --model unetpp
    python loocv.py --model attention_unet
    python loocv.py --model resunetpp
    python loocv.py --all                  # all 3 models, N*3 training runs

Output files (in config.OUTPUT_DIR)
-------------------------------------
    loocv_results_<model>.csv             per-fold scores
    loocv_boxplot_<model>.png             box plot for one model
    loocv_comparison_all_models.png       side-by-side comparison
    loocv_summary.csv                     mean ± std for every model/metric
"""

import os
import sys
import csv
import time
import argparse
import copy

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from dataset import (
    load_tif_image,
    preprocess_image,
    get_image_paths,
    _tf_load_and_label,
    _tf_augment,
)
from model_builder import build_and_compile, is_multi_output, num_outputs


# =========================================================================
# FOLD DATASET BUILDER
# Mirrors get_datasets() from dataset.py exactly but accepts explicit
# train_paths / val_paths lists instead of computing the split internally.
# =========================================================================

def get_fold_datasets(train_paths: list, val_path: str,
                       batch_size: int = None, seed: int = None):
    """
    Build tf.data train / val datasets for one LOOCV fold.

    Parameters
    ----------
    train_paths : list of str
        Paths to the N-1 training images for this fold.
    val_path    : str
        Path to the single held-out validation / test image.
    batch_size  : int, optional
    seed        : int, optional

    Returns
    -------
    train_ds, val_ds, n_train
    """
    batch_size = config.BATCH_SIZE  if batch_size is None else batch_size
    seed       = config.RANDOM_SEED if seed       is None else seed
    n_train    = len(train_paths)

    # ── Training dataset (same cache → repeat → augment pattern as
    #    get_datasets() so that LOOCV training is identical in every
    #    way to the normal training run) ───────────────────────────────
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(train_paths)
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(config.AUGMENT_REPEATS)
        .shuffle(
            buffer_size=n_train * config.AUGMENT_REPEATS,
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Validation dataset (one image, no augmentation) ─────────────
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices([val_path])
        .map(_tf_load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, n_train


# =========================================================================
# TARGET DUPLICATION FOR UNET++ DEEP SUPERVISION
# =========================================================================

def _duplicate_targets(ds, n_outputs):
    """
    Reshape (image, mask) -> (image, (mask, mask, ...)) with n_outputs
    copies of the mask, matching UNet++'s multi-output structure.
    """
    return ds.map(
        lambda img, mask: (img, tuple(mask for _ in range(n_outputs))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(pred: np.ndarray, label: np.ndarray,
                    eps: float = 1e-6) -> dict:
    """Dice, F1, IoU, Precision, Recall for two binary masks."""
    p  = pred.astype(bool).flatten()
    g  = label.astype(bool).flatten()
    tp = np.sum(p & g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou       = (tp + eps)     / (tp + fp + fn + eps)

    return dict(
        dice=float(dice), f1=float(f1), iou=float(iou),
        precision=float(precision), recall=float(recall),
        tp=int(tp), fp=int(fp), fn=int(fn),
    )


def predict_single(model, model_name: str, img: np.ndarray) -> np.ndarray:
    """Run one forward pass; return binary (H,W) uint8 mask."""
    tensor  = img[np.newaxis, ..., np.newaxis].astype(np.float32)
    outputs = model(tensor, training=False)
    logits  = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    prob    = tf.sigmoid(logits).numpy().squeeze()
    return (prob >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)


# =========================================================================
# SINGLE FOLD TRAINING
# =========================================================================

def train_one_fold(model_name: str, train_paths: list,
                   val_path: str, fold_idx: int,
                   total_folds: int) -> dict:
    """
    Train a fresh model on train_paths, evaluate on val_path.
    Returns metrics dict for this fold.
    """
    # ── Build fresh datasets ─────────────────────────────────────────
    train_ds, val_ds, n_train = get_fold_datasets(
        train_paths, val_path,
        batch_size=config.BATCH_SIZE,
        seed=config.RANDOM_SEED,
    )

    # ── Build and compile a fresh model ─────────────────────────────
    #    (build_and_compile always constructs a new instance, never
    #     reuses weights from a previous fold)
    model = build_and_compile(model_name)

    if is_multi_output(model_name):
        n_out    = num_outputs(model_name)
        train_ds = _duplicate_targets(train_ds, n_out)
        val_ds   = _duplicate_targets(val_ds,   n_out)

    # ── Monitor metric name (same logic as train.py) ──────────────
    if is_multi_output(model_name):
        monitor = f"val_output_{config.DEPTH}_dice_metric"
    else:
        monitor = "val_dice_metric"

    # ── Callbacks: EarlyStopping + ReduceLROnPlateau only
    #    No ModelCheckpoint per fold -- we only need the final weights
    #    of the best epoch, which EarlyStopping restores automatically.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max",
            factor=0.5, patience=7,
            min_lr=1e-6, verbose=0,
        ),
    ]

    steps_per_epoch = max(1, (n_train * config.AUGMENT_REPEATS) // config.BATCH_SIZE)

    print(
        f"    Fold {fold_idx}/{total_folds}  "
        f"train={n_train} imgs ({steps_per_epoch} steps/epoch)  "
        f"val=1 img  "
        f"max_epochs={config.NUM_EPOCHS}"
    )

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.NUM_EPOCHS,
        callbacks=callbacks,
        verbose=0,          # suppress per-epoch output -- fold summary is enough
    )
    elapsed = time.time() - t0

    # ── Evaluate on the held-out image ───────────────────────────────
    val_name = os.path.basename(val_path)
    raw      = load_tif_image(val_path)
    img      = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    # Reference label from Frangi pipeline
    label = preprocess_image(img)["edge_mask"]

    # CNN prediction
    pred = predict_single(model, model_name, img)

    metrics = compute_metrics(pred, label)
    metrics["image"]     = val_name
    metrics["fold"]      = fold_idx
    metrics["label_px"]  = int(label.sum())
    metrics["pred_px"]   = int(pred.sum())
    metrics["train_sec"] = round(elapsed, 1)

    print(
        f"         → Dice={metrics['dice']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"IoU={metrics['iou']:.4f}  "
        f"label={metrics['label_px']}px  "
        f"pred={metrics['pred_px']}px  "
        f"time={elapsed:.0f}s"
    )

    # Free GPU/CPU memory before next fold
    tf.keras.backend.clear_session()
    del model

    return metrics


# =========================================================================
# FULL LOOCV FOR ONE MODEL
# =========================================================================

def run_loocv(model_name: str) -> pd.DataFrame:
    """
    Run Leave-One-Out CV for one model.
    Trains N fresh models (N = number of images).
    Returns a DataFrame with one row per fold.
    """
    reg = config.MODEL_REGISTRY[model_name]
    print(f"\n{'='*64}")
    print(f"  TRUE LOOCV  →  {reg['display_name']}")
    print(f"{'='*64}")

    all_paths   = get_image_paths()
    N           = len(all_paths)
    print(f"  Total images : {N}  →  {N} folds, {N} training runs\n")

    rows = []
    for i, val_path in enumerate(all_paths):
        train_paths = [p for p in all_paths if p != val_path]
        metrics     = train_one_fold(
            model_name, train_paths, val_path,
            fold_idx=i + 1, total_folds=N,
        )
        rows.append(metrics)

    df = pd.DataFrame(rows)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  LOOCV Summary  —  {model_name}  (N={N} folds)")
    print(f"  {'─'*60}")
    for metric in ("dice", "f1", "iou", "precision", "recall"):
        v = df[metric].values
        print(
            f"  {metric:<12s}: "
            f"mean={v.mean():.4f}  std={v.std():.4f}  "
            f"min={v.min():.4f}  max={v.max():.4f}"
        )
    print(f"  {'─'*60}\n")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(
        config.OUTPUT_DIR, f"loocv_results_{model_name}.csv"
    )
    col_order = [
        "fold", "image", "label_px", "pred_px",
        "dice", "f1", "iou", "precision", "recall",
        "tp", "fp", "fn", "train_sec",
    ]
    df[col_order].to_csv(csv_path, index=False)
    print(f"  Results CSV   : {csv_path}")

    # ── Box plot ─────────────────────────────────────────────────────
    _save_boxplot(df, model_name)

    return df


# =========================================================================
# PLOTTING
# =========================================================================

def _save_boxplot(df: pd.DataFrame, model_name: str):
    """Box plot of Dice/F1/IoU/Precision/Recall across LOOCV folds."""
    metrics = ["dice", "f1", "iou", "precision", "recall"]
    labels  = ["Dice", "F1", "IoU", "Precision", "Recall"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    data    = [df[m].values for m in metrics]
    N       = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    bps = ax.boxplot(
        data, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
        widths=0.45,
    )
    for patch, color in zip(bps["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color="black", s=35, zorder=5, alpha=0.85,
            label="per-fold score" if i == 1 else "",
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"True LOOCV — {config.MODEL_REGISTRY[model_name]['display_name']}\n"
        f"N={N} folds  |  "
        f"Dice {df['dice'].mean():.3f}±{df['dice'].std():.3f}  |  "
        f"F1 {df['f1'].mean():.3f}±{df['f1'].std():.3f}  |  "
        f"IoU {df['iou'].mean():.3f}±{df['iou'].std():.3f}",
        fontsize=11,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(
        config.OUTPUT_DIR, f"loocv_boxplot_{model_name}.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Box plot      : {path}")


def save_comparison_boxplot(all_dfs: dict):
    """Side-by-side Dice/F1/IoU comparison for all three models."""
    metrics     = ["dice", "f1", "iou"]
    metric_lbls = ["Dice", "F1", "IoU"]
    model_names = list(all_dfs.keys())
    model_lbls  = [
        config.MODEL_REGISTRY[m]["display_name"].split("(")[0].strip()
        for m in model_names
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5 * len(metrics), 5), sharey=True
    )

    for ax, metric, mlbl in zip(axes, metrics, metric_lbls):
        data = [all_dfs[m][metric].values for m in model_names]
        bps  = ax.boxplot(
            data, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.45,
        )
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(data, start=1):
            jitter = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color="black", s=30, zorder=5, alpha=0.85,
            )

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_lbls, fontsize=9, rotation=15, ha="right")
        ax.set_title(mlbl, fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=12)

    fig.suptitle(
        "True LOOCV Comparison — All Three Models",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(
        config.OUTPUT_DIR, "loocv_comparison_all_models.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot : {path}")


def print_summary_table(all_dfs: dict):
    """Print and save a clean mean±std table for all models."""
    metrics = ["dice", "f1", "iou", "precision", "recall"]
    lbls    = ["Dice", "F1", "IoU", "Prec", "Recall"]

    print(f"\n{'='*72}")
    print("  TRUE LOOCV FINAL SUMMARY  —  mean ± std")
    print(f"{'='*72}")
    print(f"  {'Model':<30s}" + "".join(f"{l:>12s}" for l in lbls))
    print(f"  {'-'*68}")

    summary_rows = []
    for mname, df in all_dfs.items():
        disp = config.MODEL_REGISTRY[mname]["display_name"][:29]
        row  = f"  {disp:<30s}"
        r    = {"model": mname}
        for metric in metrics:
            v = df[metric].values
            row += f"  {v.mean():.3f}±{v.std():.3f}"
            r[f"{metric}_mean"] = round(float(v.mean()), 4)
            r[f"{metric}_std"]  = round(float(v.std()),  4)
        print(row)
        summary_rows.append(r)

    print(f"{'='*72}\n")

    path = os.path.join(config.OUTPUT_DIR, "loocv_summary.csv")
    pd.DataFrame(summary_rows).to_csv(path, index=False)
    print(f"  Summary CSV : {path}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="True LOOCV: retrain fresh model per fold."
    )
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=list(config.MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run LOOCV for all three models.",
    )
    args = parser.parse_args()

    if args.all:
        all_dfs = {}
        for name in config.MODEL_REGISTRY.keys():
            all_dfs[name] = run_loocv(name)

        if len(all_dfs) > 1:
            save_comparison_boxplot(all_dfs)

        print_summary_table(all_dfs)
    else:
        run_loocv(args.model)
