"""
main.py - ablation study runner (LOOCV by default), one model per run:

    python main.py --model unet
    python main.py --model resse --match-params --filters 8
    python main.py --model encdec attunet          # several models in one run also works

Where things live:
    args.py                    command-line options  (the object `a`)
    dataset.py                 all data handling
    models/models.py           all models (get_model)
    utils/ablation_losses.py   loss
    utils/ablation_metrics.py  metrics, tables, statistics
    utils/ablation_plots.py    all figures
    utils/ablation_io.py       run folders, log, CSVs, predictions
Every run is saved in its OWN folder  <OUTPUT_DIR>/ablation/<timestamp>_<name>/
"""
from headers import *                    # your common imports (kept)

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from args import a

from dataset import (load_all_images, make_folds, check_sizes, make_patches,
                     get_repeats, build_train_ds_from_arrays)
import inspect
import models.models as M
get_model = M.get_model
count_parameters = getattr(M, "count_parameters", None) or (lambda m: int(m.count_params()))
ALL_MODELS = getattr(M, "ALL_MODELS", None)   # None: your models.py has no list, names are not checked here

from utils.ablation_losses import bce_dice_loss
from utils.ablation_metrics import (METRICS, confusion_counts, metrics_from_counts,
                                    summarize, stats_on_dice, label)
from utils.ablation_plots import plot_violins, plot_prediction_grids, plot_loss_curves
from utils.ablation_io import (make_run_dir, start_log, append_rows, save_json,
                               save_fold_predictions)

BASE_FILTERS = getattr(config, "BASE_FILTERS", 16)
DEPTH = getattr(config, "DEPTH", 4)
VARIANT_FILTERS = {}      # per-model base filters, filled by --match-params


# ----------------------------------------------------------------- training
def build_model(variant, base_filters):
    """Calls get_model whatever its signature is:
       get_model(variant, input_shape, base_filters, depth)   (models/models.py from me), or
       get_model(input_shape, base_filters, depth)            (the model is chosen through `a.model`)."""
    kw = dict(input_shape=(None, None, 1), base_filters=base_filters, depth=DEPTH)
    params = list(inspect.signature(get_model).parameters)
    for name in ("variant", "name", "model_name", "model", "arch"):
        if name in params:
            return get_model(**{name: variant}, **kw)
    if params and params[0] not in kw:                 # name is the first, differently named parameter
        return get_model(variant, **kw)
    a.model = [variant] if isinstance(a.model, list) else variant
    return get_model(**kw)


def train(variant, X_tr, Y_tr, epochs, seed, full_images=False, verbose_info=False):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    Xp, Yp = (X_tr, Y_tr) if full_images else make_patches(X_tr, Y_tr)
    repeats = get_repeats(full_images)
    if verbose_info:
        steps = int(np.ceil(len(Xp) * repeats / config.BATCH_SIZE))
        print(f"  training on {'full images' if full_images else 'patches'}: {len(Xp)} of "
              f"shape {Xp.shape[1:3]}, repeats={repeats}, ~{steps} steps/epoch")

    # free spatial size: train on patches, predict on the full image
    model = build_model(variant, VARIANT_FILTERS.get(variant, BASE_FILTERS))
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss=bce_dice_loss)
    history = model.fit(
        build_train_ds_from_arrays(Xp, Yp, seed=seed, repeats=repeats),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0)],
        verbose=2,
    )
    return model, history.history          # history: loss (and learning rate) per epoch


def loss_rows_from_history(variant, fold, hist):
    """One row per epoch: model, fold, epoch, training loss (+ learning rate if logged)."""
    lr = hist.get("learning_rate", hist.get("lr"))
    rows = []
    for e, loss in enumerate(hist["loss"]):
        row = dict(model=variant, fold=fold, epoch=e + 1, loss=float(loss))
        if lr is not None:
            row["learning_rate"] = float(lr[e])
        rows.append(row)
    return rows


def predict_mask(model, img):
    """img: one image (H, W, 1) or a batch of one (1, H, W, 1) -> binary mask (H, W)."""
    if img.ndim == 3:
        img = img[None]
    logits = model(img, training=False)
    prob = tf.sigmoid(logits).numpy().squeeze()
    return (prob >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)


def match_filters(models, ref_filters, ref="unet"):
    """Parameter matching: `ref` (unet) uses ref_filters; every other model gets the
    base_filters (a float, found by bisection) that gives it the SAME number of parameters."""
    def n_params(v, f):
        tf.keras.backend.clear_session()
        return count_parameters(build_model(v, f))

    target = n_params(ref, ref_filters)
    out = {ref: ref_filters}
    print(f"\nParameter matching: target = {target:,} parameters ({ref} with base_filters={ref_filters})")
    for v in models:
        if v == ref:
            continue
        lo, hi = 2.0, 64.0
        for _ in range(10):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if n_params(v, mid) < target else (lo, mid)
        f = min((lo, hi), key=lambda x: abs(n_params(v, x) - target))
        out[v] = round(f, 3)
        n = n_params(v, out[v])
        print(f"  {v:<8s} base_filters={out[v]:<7} -> {n:,} parameters ({(n - target) / target:+.1%})")
    tf.keras.backend.clear_session()
    return out


# ----------------------------------------------------------------- main run
def run(epochs, k, seed, models, full_images, match_params, ref, run_dir, cli_args):
    check_sizes(DEPTH, full_images)

    if isinstance(models, str):                       # --model unet  ->  ["unet"]
        models = [models]
    aliases = getattr(M, "ALIASES", {})               # e.g. attunet_aspp -> attaspp
    models = [aliases.get(m.lower(), m.lower()) for m in (models or [])] or None
    if ALL_MODELS is not None:
        unknown = [m for m in (models or []) if m not in ALL_MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. Available in models/models.py: {ALL_MODELS}")
        to_train = [m for m in ALL_MODELS if models is None or m in models]
    else:
        to_train = list(models)
    print(f"\nModels: {', '.join(to_train)}")
    if match_params:
        VARIANT_FILTERS.update(match_filters(to_train, BASE_FILTERS))

    names, X, Y = load_all_images()
    n = len(names)
    folds = make_folds(n, k, seed)
    k = len(folds)
    kind = "LOOCV" if k == n else f"shuffled {k}-fold CV"
    print(f"{n} images, {kind} ({k} folds, seed={seed}), {epochs} epochs/fold, "
          f"{'full images' if full_images else 'patches'}")
    for f, idx in enumerate(folds, 1):
        print(f"  fold {f}: test = {[names[i] for i in idx]}")

    # everything needed to reproduce / identify this run
    meta = dict(
        command=" ".join(sys.argv), args=cli_args, epochs=epochs, seed=seed,
        cv=kind, n_folds=k, trained_models=to_train, stats_reference=ref,
        images=names, folds_test=[[names[i] for i in idx] for idx in folds],
        training_on="full images" if full_images else "patches",
        base_filters=BASE_FILTERS, variant_filters=VARIANT_FILTERS, depth=DEPTH,
        parameters={},
        image_size=config.IMAGE_SIZE, patch_size=config.PATCH_SIZE,
        patch_stride=config.PATCH_STRIDE, batch_size=config.BATCH_SIZE,
        augment_repeats=config.AUGMENT_REPEATS, learning_rate=config.LEARNING_RATE,
        prob_threshold=config.SEGMENTATION_PROB_THRESHOLD,
    )
    save_json(os.path.join(run_dir, "run_config.json"), meta)

    per_image_csv = os.path.join(run_dir, "ablation_per_image.csv")
    loss_csv = os.path.join(run_dir, "ablation_loss_per_epoch.csv")
    loss_rows = []
    rows, preds = [], {m: {} for m in to_train}
    for f, test_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        fold_preds = {}
        for mi, m in enumerate(to_train):
            print(f"\n[fold {f}/{k}] {label(m)}  "
                  f"(train {len(train_idx)}, test {len(test_idx)})")
            model, hist = train(m, X[train_idx], Y[train_idx], epochs, seed,
                          full_images=full_images, verbose_info=(f == 1 and mi == 0))
            if f == 1:
                meta["parameters"][m] = count_parameters(model)
                print(f"  parameters: {meta['parameters'][m]:,}")
                save_json(os.path.join(run_dir, "run_config.json"), meta)
            new_loss = loss_rows_from_history(m, f, hist)
            loss_rows += new_loss
            append_rows(loss_csv, new_loss)                  # loss per epoch, saved immediately
            new_rows = []
            for i in test_idx:
                pred = predict_mask(model, X[i])
                preds[m][i] = pred
                fold_preds[f"{m}__{names[i]}"] = pred
                c = confusion_counts(pred, Y[i, ..., 0])
                row = dict(model=m, fold=f, image=names[i], img_idx=int(i),
                           **c, **metrics_from_counts(**c))
                new_rows.append(row)
                print(f"  {names[i]:<18s} Dice={row['dice']:.4f}  IoU={row['iou']:.4f}  "
                      f"P={row['precision']:.3f}  R={row['recall']:.3f}  Acc={row['accuracy']:.4f}")
            rows += new_rows
            append_rows(per_image_csv, new_rows)         # saved immediately
            del model
            tf.keras.backend.clear_session()
        save_fold_predictions(run_dir, f, fold_preds)    # masks of this fold

    df_img = pd.DataFrame(rows)
    df_fold = df_img.groupby(["model", "fold"], as_index=False)[METRICS].mean()
    df_fold.to_csv(os.path.join(run_dir, "ablation_per_fold.csv"), index=False)

    summarize(df_fold, df_img, to_train, run_dir)
    if len(to_train) >= 2:                      # statistics need 2+ models
        stats_on_dice(df_fold, to_train, run_dir, ref=ref)
    plot_violins(df_fold, to_train, run_dir, n_images=n)
    plot_loss_curves(pd.DataFrame(loss_rows), to_train, run_dir)
    plot_prediction_grids(names, X, Y, preds, df_img, to_train, run_dir)
    print(f"\nAll outputs of this run saved in: {run_dir}")


def run_name(a, seed):
    """Readable run-folder name built from the options."""
    models = [a.model] if isinstance(a.model, str) else (a.model or [])
    parts = ["-".join(models)]
    if a.match_params:
        parts.append("match")
    if getattr(a, "filters", None):
        parts.append(f"f{a.filters}")
    parts.append("full" if a.full_images else f"p{getattr(a, 'patch', None) or 'cfg'}")
    parts.append(f"k{a.folds}" if a.folds else "loocv")
    parts.append(f"s{seed}")
    return "_".join(parts)


if __name__ == "__main__":
    # options that may not exist in an older args.py are read with getattr
    if getattr(a, "light", False):                       # --light = --match-params --filters 8
        a.match_params = True
        a.filters = getattr(a, "filters", None) or 8
    if getattr(a, "filters", None):
        BASE_FILTERS = a.filters
    if getattr(a, "patch", None):
        config.PATCH_SIZE = a.patch
        config.PATCH_STRIDE = a.patch // 2

    seed = config.RANDOM_SEED if a.seed is None else a.seed
    epochs = a.epochs or config.NUM_EPOCHS
    ref = getattr(a, "ref", None) or "unet"

    run_dir = make_run_dir(os.path.join(config.OUTPUT_DIR, "ablation"),
                           getattr(a, "name", None) or run_name(a, seed))
    start_log(run_dir)
    print(f"Run folder: {run_dir}")

    run(epochs, a.folds, seed, a.model, a.full_images, a.match_params, ref, run_dir, vars(a))