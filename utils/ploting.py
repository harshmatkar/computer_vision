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
        f"LOOCV Results — {model_name}\n"
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