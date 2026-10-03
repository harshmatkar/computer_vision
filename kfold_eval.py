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
matplotlib.use("Agg")          # non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt
import pandas as pd

# ── Make sure the project root is importable ─────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from dataset import load_tif_image, preprocess_image, get_image_paths
from model_builder import is_multi_output


# =========================================================================
# HELPERS
# =========================================================================

def load_model(model_name: str):
    """Load a trained .keras checkpoint for the given model name."""
    ckpt_name = config.MODEL_REGISTRY[model_name]["checkpoint_name"]
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run:  python train.py --model {model_name}"
        )
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


def compute_metrics(pred: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> dict:
    """
    Compute Dice, F1, IoU, Precision, Recall between a predicted binary
    mask and a reference binary mask (the Frangi-HOG auto-label).
    """
    p = pred.astype(bool).flatten()
    g = label.astype(bool).flatten()

    tp = np.sum(p & g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = (2 * precision * recall) / (precision + recall + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)

    return {
        "dice":      float(dice),
        "f1":        float(f1),
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
        "tp":        int(tp),
        "fp":        int(fp),
        "fn":        int(fn),
    }


# =========================================================================
# LOOCV
# =========================================================================

def run_loocv(model_name: str) -> pd.DataFrame:
    """
    Run Leave-One-Out CV for one model.

    Returns a DataFrame with one row per image containing all metrics.
    Also saves:
        outputs/loocv_results_<model_name>.csv
        outputs/loocv_boxplot_<model_name>.png
    """
    print(f"\n{'='*60}")
    print(f"LOOCV  →  {config.MODEL_REGISTRY[model_name]['display_name']}")
    print(f"{'='*60}")

    model       = load_model(model_name)
    image_paths = get_image_paths()
    N           = len(image_paths)
    print(f"  Images  : {N}  (each treated as held-out test once)\n")

    rows = []

    for i, path in enumerate(image_paths):
        name = os.path.basename(path)

        # ── Load & resize ─────────────────────────────────────────────
        raw = load_tif_image(path)
        img = cv2.resize(
            raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
        )

        # ── Generate Frangi label (reference) ─────────────────────────
        result = preprocess_image(img)
        label  = result["edge_mask"]

        # ── CNN prediction ─────────────────────────────────────────────
        pred = predict(model, model_name, img)

        # ── Metrics ────────────────────────────────────────────────────
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

    # ── Summary statistics ─────────────────────────────────────────────
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

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(
        config.OUTPUT_DIR, f"loocv_results_{model_name}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  Results saved : {csv_path}")

    # ── Box plot ───────────────────────────────────────────────────────
    _save_boxplot(df, model_name)

    return df


# =========================================================================
# BOX PLOT
# =========================================================================

def _save_boxplot(df: pd.DataFrame, model_name: str):
    """
    Save a box-plot figure showing the distribution of Dice, F1, IoU,
    Precision and Recall across LOOCV folds for one model.
    """
    metrics  = ["dice", "f1", "iou", "precision", "recall"]
    data     = [df[m].values for m in metrics]
    labels   = ["Dice", "F1", "IoU", "Precision", "Recall"]
    colors   = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bps = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
        widths=0.45,
    )
    for patch, color in zip(bps["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual data points (one dot per image/fold)
    for i, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color="black",
            s=30,
            zorder=5,
            alpha=0.8,
            label="per-image score" if i == 1 else "",
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"LOOCV Results — {config.MODEL_REGISTRY[model_name]['display_name']}\n"
        f"N={len(df)} images  │  "
        f"Dice  {df['dice'].mean():.3f} ± {df['dice'].std():.3f}  │  "
        f"F1  {df['f1'].mean():.3f} ± {df['f1'].std():.3f}  │  "
        f"IoU  {df['iou'].mean():.3f} ± {df['iou'].std():.3f}",
        fontsize=11,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(
        config.OUTPUT_DIR, f"loocv_boxplot_{model_name}.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Box plot saved: {plot_path}")


# =========================================================================
# ALL-MODELS COMPARISON BOX PLOT
# =========================================================================

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


# =========================================================================
# SUMMARY TABLE
# =========================================================================

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


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-One-Out Cross-Validation evaluation."
    )
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=list(config.MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run LOOCV for all three models and save a comparison plot.",
    )
    args = parser.parse_args()

    if args.all:
        all_dfs = {}
        for name in config.MODEL_REGISTRY.keys():
            try:
                all_dfs[name] = run_loocv(name)
            except FileNotFoundError as e:
                print(f"\n  Skipping {name}: {e}")

        if len(all_dfs) > 1:
            save_comparison_boxplot(all_dfs)

        if all_dfs:
            print_summary_table(all_dfs)
    else:
        run_loocv(args.model)
