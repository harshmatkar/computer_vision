"""
utils/ablation_metrics.py
-------------------------
Metrics, summary tables and statistics for the ablation study.
No plotting (see ablation_plots.py) and no data loading (see dataset.py).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats


# Display names. Models that are not listed simply use their own name.
MODEL_LABELS = {
    "encdec": "Enc-Dec", "unet": "U-Net", "resunet": "ResUNet", "resse": "ResUNet+SE",
    "attunet": "Att-UNet", "attaspp": "Att-UNet+ASPP",
}


def label(m):
    return MODEL_LABELS.get(m, m)

METRICS = ["dice", "iou", "precision", "recall", "accuracy"]


# ----------------------------------------------------------------- metrics
def confusion_counts(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    return dict(tp=int((p & g).sum()), fp=int((p & ~g).sum()),
                fn=int((~p & g).sum()), tn=int((~p & ~g).sum()))


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-8
    return dict(
        dice=2 * tp / (2 * tp + fp + fn + eps),
        iou=tp / (tp + fp + fn + eps),
        precision=tp / (tp + fp + eps),
        recall=tp / (tp + fn + eps),
        accuracy=(tp + tn) / (tp + fp + fn + tn + eps),
    )


# ----------------------------------------------------------------- tables
def _name(m):
    return label(m)


def summarize(df_fold, df_img, models, out_dir):
    """Print + save mean/std over folds and pooled confusion counts."""
    print(f"\n{'='*92}\n  MEAN +/- STD OVER FOLDS (each fold = one value)\n{'='*92}")
    print(f"  {'Model':<18s}" + "".join(f"{m:>16s}" for m in METRICS))
    rows = []
    for m in models:
        sub = df_fold[df_fold.model == m]
        line, r = f"  {_name(m):<18s}", {"model": m}
        for met in METRICS:
            v = sub[met].values
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            line += f"  {v.mean():.3f}+/-{sd:.3f}"
            r[f"{met}_mean"], r[f"{met}_std"] = round(v.mean(), 4), round(sd, 4)
        print(line)
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)

    print(f"\n{'='*92}\n  POOLED CONFUSION COUNTS (all test images) AND SEGMENTATION BEHAVIOUR\n{'='*92}")
    print(f"  {'Model':<18s}{'TP':>10s}{'FP':>10s}{'FN':>10s}{'TN':>12s}{'FP/FN':>8s}  Behaviour")
    conf = []
    for m in models:
        s = df_img[df_img.model == m][["tp", "fp", "fn", "tn"]].sum()
        ratio = s.fp / max(s.fn, 1)
        if ratio > 1.2:
            verdict = "over-segmentation (FP > FN)"
        elif ratio < 1 / 1.2:
            verdict = "under-segmentation (FN > FP)"
        else:
            verdict = "balanced"
        print(f"  {_name(m):<18s}{int(s.tp):>10d}{int(s.fp):>10d}"
              f"{int(s.fn):>10d}{int(s.tn):>12d}{ratio:>8.2f}  {verdict}")
        conf.append(dict(model=m, tp=int(s.tp), fp=int(s.fp), fn=int(s.fn),
                         tn=int(s.tn), fp_over_fn=round(ratio, 3), behaviour=verdict))
    pd.DataFrame(conf).to_csv(os.path.join(out_dir, "ablation_confusion.csv"), index=False)
    print("\n  Note: accuracy is dominated by the large background class (TN), so it is")
    print("  high for every model; Dice / IoU / precision / recall are more informative.")


# ----------------------------------------------------------------- statistics
def stats_on_dice(df_fold, models, out_dir, ref="unet"):
    """Friedman test over all models + paired tests of every model against the
    reference model (Bonferroni over those comparisons)."""
    if len(models) < 2:
        print("\n  Statistics skipped: only one model was trained.")
        return
    d = {m: df_fold[df_fold.model == m].sort_values("fold")["dice"].values for m in models}
    n = len(next(iter(d.values())))
    res = []
    print(f"\n{'='*92}\n  STATISTICS ON DICE (paired over {n} folds)\n{'='*92}")
    if len(models) >= 3:
        fs, fp = stats.friedmanchisquare(*[d[m] for m in models])
        print(f"  Friedman test (all models): statistic = {fs:.4f}, p = {fp:.4f}")
        res.append(dict(test="Friedman (overall)", comparison="all models",
                        statistic=fs, p_raw=fp, p_adj=np.nan))
    if ref not in d:
        ref = models[0]
    pairs = [(ref, m) for m in models if m != ref]
    print(f"  Every model compared with the reference '{ref}':")
    for a, b in pairs:
        t, pt = stats.ttest_rel(d[a], d[b])
        try:
            w, pw = stats.wilcoxon(d[a], d[b])
        except ValueError:
            w, pw = np.nan, np.nan
        adj = lambda p: min(1.0, p * len(pairs)) if p == p else np.nan
        print(f"  {a} vs {b}:  mean diff = {np.mean(d[b]-d[a]):+.4f} | "
              f"paired t p = {pt:.4f} (adj {adj(pt):.4f}) | "
              f"Wilcoxon p = {pw:.4f} (adj {adj(pw):.4f})")
        res.append(dict(test="paired t", comparison=f"{a} vs {b}", statistic=t,
                        p_raw=pt, p_adj=adj(pt)))
        res.append(dict(test="Wilcoxon", comparison=f"{a} vs {b}", statistic=w,
                        p_raw=pw, p_adj=adj(pw)))
    print(f"  With n={n} folds the smallest possible two-sided Wilcoxon p is "
          f"{2 / 2**n:.4f}. Folds share training images, so treat p-values as indicative.")
    pd.DataFrame(res).to_csv(os.path.join(out_dir, "ablation_stats.csv"), index=False)