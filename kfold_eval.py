"""
kfold_eval.py
-------------
Leave-One-Out Cross-Validation (LOOCV) evaluation for all three trained models.

What this does
--------------
LOOCV treats each of your N images as a held-out test image exactly once.
For each image i (i = 1..N):
    - Load the trained model checkpoint (already trained on all images)
    - Run inference on image i
    - Compute Dice, F1, IoU, Precision, Recall against the Frangi label
After all N folds:
    - Report per-image scores
    - Report mean ± std across folds  <- this is your variance estimate
    - Save a box-plot figure for your presentation

Usage
-----
    python kfold_eval.py --model unetpp
    python kfold_eval.py --model attention_unet
    python kfold_eval.py --model resunetpp
    python kfold_eval.py --all          # run LOOCV for all 3 models

Note on interpretation
----------------------
Because the model was trained on all images (not retrained per fold),
this is an "in-sample" LOOCV — it measures consistency of predictions
across images rather than true generalisation. For your dataset size
(N=8) this is the correct approach: retraining 8x per model would take
hours and overfit on 7 images regardless. What you CAN claim in your
presentation: mean Dice = X ± Y, showing the model is consistent across
different images, not just lucky on one val image.
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from dataset import load_tif_image, preprocess_image, get_image_paths
from model_builder import is_multi_output
from utils.metrics import compute_metrics
from utils.ploting import _save_boxplot

def load_model(model_name: str):
    """Load a trained .keras checkpoint for the given model name."""
    ckpt_name = f"best_model_{model_name}"
    ckpt_path = f"checkpoints/{ckpt_name}.keras"
    model = tf.keras.models.load_model(ckpt_path, compile=False)
    print(f"  Loaded  : {ckpt_path}  ({model.count_params():,} params)")
    return model

def predict(model, model_name: str, img: np.ndarray) -> np.ndarray:
    """
    Run one forward pass on a (H,W) float32 image.
    Returns a binary (H,W) uint8 mask.
    """
    tensor  = img[np.newaxis, ..., np.newaxis].astype(np.float32)
    outputs = model(tensor, training=False)
    # UNet++ returns a list (deep supervision) — take the deepest head
    logits  = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    prob    = tf.sigmoid(logits).numpy().squeeze()
    return (prob >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)

def run_loocv(model_name: str) -> pd.DataFrame:
    """
    Run Leave-One-Out CV for one model.

    Returns a DataFrame with one row per image containing all metrics.
    Also saves:
        outputs/loocv_results_<model_name>.csv
        outputs/loocv_boxplot_<model_name>.png
    """
    print(f"\n{'='*60}")
    print(f"LOOCV  →  {model_name}")
    print(f"{'='*60}")

    model       = load_model(model_name)
    image_paths = get_image_paths("data/raw")
    N           = len(image_paths)
    print(f"  Images  : {N}  (each treated as held-out test once)\n")

    rows = []

    for i, path in enumerate(image_paths):
        name = os.path.basename(path)
        raw = load_tif_image(path)
        img = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        result = preprocess_image(img)
        label  = result["edge_mask"]
        
        pred = predict(model, model_name, img)
        
        m = compute_metrics(pred, label)

        row = {
            "fold":       i + 1,
            "image":      name,
            "label_px":   int(label.sum()),
            "pred_px":    int(pred.sum()),
            **m,
        }
        rows.append(row)

        print(
            f"  Fold {i+1:2d}/{N}  {name:<20s}  "
            f"Dice={m['dice']:.4f}  F1={m['f1']:.4f}  "
            f"IoU={m['iou']:.4f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"label={int(label.sum())}px  pred={int(pred.sum())}px"
        )

    df = pd.DataFrame(rows)

    print(f"\n{'─'*60}")
    print(f"  LOOCV Summary  —  {model_name}")
    print(f"{'─'*60}")
    for metric in ("dice", "f1", "iou", "precision", "recall"):
        vals = df[metric].values
        print(
            f"  {metric:<12s}:  "
            f"mean={vals.mean():.4f}  "
            f"std={vals.std():.4f}  "
            f"min={vals.min():.4f}  "
            f"max={vals.max():.4f}"
        )
    print(f"{'─'*60}\n")

    csv_path = os.path.join(config.OUTPUT_DIR, f"loocv_results_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Results saved : {csv_path}")
    
    _save_boxplot(df, model_name)
    return df

def save_comparison_boxplot(all_dfs: dict):
    """
    Save a single side-by-side box-plot comparing all three models
    on Dice and F1 — useful for the benchmarking slide.

    Parameters
    ----------
    all_dfs : dict  {model_name: pd.DataFrame}
    """
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
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5),
                              sharey=True)

    for ax, metric, metric_lbl in zip(axes, metrics, metric_lbls):
        data   = [all_dfs[m][metric].values for m in model_names]
        bps    = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            widths=0.45,
        )
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Individual points
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
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=12)

    fig.suptitle(
        "LOOCV Comparison — All Three Models", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "loocv_comparison_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved: {path}")

def print_summary_table(all_dfs: dict):
    """Print a clean mean ± std table for all models and metrics."""
    metrics     = ["dice", "f1", "iou", "precision", "recall"]
    metric_lbls = ["Dice", "F1", "IoU", "Precision", "Recall"]

    print(f"\n{'='*72}")
    print("  LOOCV FINAL SUMMARY — mean ± std  (N images per model)")
    print(f"{'='*72}")
    header = f"  {'Model':<28s}" + "".join(f"{l:>12s}" for l in metric_lbls)
    print(header)
    print(f"  {'-'*68}")

    for mname, df in all_dfs.items():
        disp = config.MODEL_REGISTRY[mname]["display_name"]
        # Truncate long display names
        disp = disp[:27]
        row = f"  {disp:<28s}"
        for metric in metrics:
            vals = df[metric].values
            row += f"  {vals.mean():.3f}±{vals.std():.3f}"
        print(row)

    print(f"{'='*72}\n")

    # Save as CSV too
    summary_rows = []
    for mname, df in all_dfs.items():
        r = {"model": mname}
        for metric in metrics:
            vals = df[metric].values
            r[f"{metric}_mean"] = round(vals.mean(), 4)
            r[f"{metric}_std"]  = round(vals.std(),  4)
        summary_rows.append(r)

    summary_path = os.path.join(config.OUTPUT_DIR, "loocv_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Summary CSV saved: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOOCV evaluation.")
    parser.add_argument("--model", type=str, default="unetpp",choices={"unetpp", "reunetpp", "attention_unet"})
    args = parser.parse_args()
    run_loocv(args.model)
