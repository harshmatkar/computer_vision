"""
utils/ablation_plots.py
-----------------------
All plotting for the ablation study: violin plots, prediction grids and
error maps. No metrics (ablation_metrics.py) and no data loading (dataset.py).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.ablation_metrics import METRICS, label

COLORS = {"encdec": "#9E9E9E", "unet": "#2196F3", "resunet": "#4CAF50",
          "resse": "#009688", "attunet": "#FF9800", "attaspp": "#E91E63"}


def plot_violins(df_fold, models, out_dir, n_images=None):
    titles = {"dice": "Dice", "iou": "IoU", "precision": "Precision",
              "recall": "Recall", "accuracy": "Accuracy"}
    n_models = len(models)
    n_folds = df_fold["fold"].nunique()
    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(max(4.2, 1.5 * n_models + 1.5) * len(METRICS), 5.6))
    rng = np.random.default_rng(0)
    for ax, met in zip(axes, METRICS):
        data = [df_fold[df_fold.model == m].sort_values("fold")[met].values for m in models]
        pos = np.arange(1, len(models) + 1)
        for p, vals, m in zip(pos, data, models):
            if np.ptp(vals) > 1e-9 and len(vals) > 2:
                vp = ax.violinplot(vals, positions=[p], widths=0.8,
                                   showmeans=False, showmedians=True, showextrema=False)
                for b in vp["bodies"]:
                    b.set_facecolor(COLORS.get(m, "#607D8B")); b.set_alpha(0.55)
                    b.set_edgecolor("black")
                vp["cmedians"].set_color("black")
        arr = np.array(data)                       # (models, folds)
        for j in range(arr.shape[1]):              # grey line = same fold across models
            ax.plot(pos, arr[:, j], color="grey", alpha=0.35, lw=0.8, zorder=2)
        for p, vals in zip(pos, data):
            ax.scatter(p + rng.uniform(-0.07, 0.07, len(vals)), vals,
                       color="black", s=22, zorder=5)
        ax.set_xticks(pos)
        ax.set_xticklabels([label(m) for m in models],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(titles[met], fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        if met != "accuracy":
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Score (one point per fold)")
    if n_images is not None and n_folds == n_images:
        title = ("Ablation study (LOOCV, one point per test image; "
                 "grey lines join the same image)")
    else:
        title = (f"Ablation study (shuffled {n_folds}-fold CV; one point per fold, "
                 f"grey lines join the same fold)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_violins.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Violin plot saved: {path}")


def plot_prediction_grids(names, X, Y, preds, df_img, models, out_dir):
    n = len(names)
    # ---- figure 1: image | ground truth | predictions
    ncols = 2 + len(models)
    fig, axes = plt.subplots(n, ncols, figsize=(2.8 * ncols, 2.8 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(X[i, ..., 0], cmap="gray")
        axes[i, 0].set_title(names[i] if i else f"Image\n{names[i]}", fontsize=8)
        axes[i, 1].imshow(Y[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground truth" if i == 0 else "", fontsize=9)
        for c, m in enumerate(models, start=2):
            dsc = df_img[(df_img.model == m) & (df_img.img_idx == i)]["dice"].iloc[0]
            axes[i, c].imshow(preds[m][i], cmap="gray", vmin=0, vmax=1)
            axes[i, c].set_title(f"{label(m)}\nDice={dsc:.3f}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "ablation_predictions.png")
    plt.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()

    # ---- figure 2: error maps (TP green, FP red, FN blue)
    ncols = 1 + len(models)
    fig, axes = plt.subplots(n, ncols, figsize=(2.8 * ncols, 2.8 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(X[i, ..., 0], cmap="gray")
        axes[i, 0].set_title(names[i], fontsize=8)
        g = Y[i, ..., 0].astype(bool)
        for c, m in enumerate(models, start=1):
            p = preds[m][i].astype(bool)
            rgb = np.zeros(g.shape + (3,), np.float32)
            rgb[p & g] = (0.1, 0.8, 0.1)       # TP green
            rgb[p & ~g] = (0.9, 0.1, 0.1)      # FP red   -> over-segmentation
            rgb[~p & g] = (0.2, 0.4, 1.0)      # FN blue  -> under-segmentation
            axes[i, c].imshow(rgb)
            fp, fn = int((p & ~g).sum()), int((~p & g).sum())
            axes[i, c].set_title(f"{label(m)}\nFP={fp}  FN={fn}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Green = TP   Red = FP (over-segmentation)   Blue = FN (under-segmentation)   "
                 "Black = TN", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    p2 = os.path.join(out_dir, "ablation_errors.png")
    plt.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Prediction grid saved: {p1}\n  Error-map grid saved: {p2}")