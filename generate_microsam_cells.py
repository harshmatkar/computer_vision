"""
generate_microsam_cells.py
--------------------------
Segment the CELLS in every image with micro_sam (Segment Anything for
Microscopy; Archit et al., Nature Methods 22:579-591, 2025) and save them as
integer label images that the actin labeler can use instead of its own
Otsu + watershed cells.

Run this in an environment that has micro_sam installed (separate from your
TensorFlow environment, e.g. a new conda env or Google Colab). It does not
need TensorFlow, OpenCV, or any other file of this project.

    python generate_microsam_cells.py --input data/raw --output data/cells_microsam

Outputs (in --output)
    <stem>_cells.tif          uint16 label image, 0 = background, 1..N = cells,
                              same size as the input image
    <stem>_cells_preview.png  input image with the cell borders in red - LOOK
                              AT THESE before trusting the masks
    cells_manifest.json       model, mode, thresholds, versions, per-image counts

Then, in config.py:   CELL_SOURCE = "microsam"   (and LABEL_MODE = "statement" or "union")

What this produces is a CELL segmentation, not an actin mask. The actin label
is still made by label_statement.py: per-cell boundary band + HOG + local
contrast threshold, now computed on micro_sam's cells.

Model choices (light microscopy): vit_b_lm (default, good balance),
vit_l_lm (larger, slower), vit_t_lm (smallest, fastest).
"""

import os
import sys
import json
import glob
import time
import inspect
import argparse

import numpy as np
from PIL import Image


# =========================================================================
# Pure helpers (no micro_sam needed - unit-testable)
# =========================================================================

def load_image(path: str) -> np.ndarray:
    """Load an image (16-bit / multi-channel ok) as float32 in [0,1]; first channel, min-max.
    Same convention as the training pipeline's load_tif_image()."""
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    lo, hi = float(arr.min()), float(arr.max())
    return ((arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)).astype(np.float32)


def to_uint8(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.8) -> np.ndarray:
    """
    Contrast-stretch to uint8 between two percentiles. Fluorescence images often
    have a few extremely bright pixels; micro_sam min-max normalises its input, so
    without this the rest of the image would look dark to the model.
    Use (0, 100) to disable the clipping.
    """
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def clean_instances(labels: np.ndarray, min_px: int = 50) -> np.ndarray:
    """Drop cells smaller than min_px, make ids consecutive 1..N, return uint16."""
    lab = np.asarray(labels).astype(np.int64)
    lab[lab < 0] = 0
    if lab.max() > 0 and min_px > 0:
        counts = np.bincount(lab.ravel())
        small = counts < min_px
        small[0] = False
        lab[small[lab]] = 0
    uniq, inv = np.unique(lab, return_inverse=True)
    lab = inv.reshape(lab.shape)
    if uniq[0] != 0:
        lab = lab + 1
    if lab.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"{int(lab.max())} cells do not fit in uint16 - something is wrong with the segmentation")
    return lab.astype(np.uint16)


def summarize(labels: np.ndarray) -> dict:
    n = int(labels.max())
    areas = np.bincount(labels.ravel())[1:] if n > 0 else np.array([0])
    return {
        "n_cells": n,
        "foreground_percent": round(float((labels > 0).mean() * 100), 2),
        "median_cell_area_px": int(np.median(areas)) if n > 0 else 0,
    }


def sanity_warnings(stats: dict) -> list:
    w = []
    if stats["n_cells"] < 5:
        w.append("fewer than 5 cells found - the segmentation probably failed; inspect the preview")
    if stats["foreground_percent"] > 98:
        w.append("cells cover >98% of the image - background not separated")
    if stats["foreground_percent"] < 10:
        w.append("cells cover <10% of the image - most of the tissue was missed")
    return w


def save_cells(labels: np.ndarray, path: str) -> None:
    """Save uint16 labels as an UNcompressed tif, readable by any tifffile/PIL install."""
    import tifffile
    tifffile.imwrite(path, labels.astype(np.uint16))


def save_preview(img_u8: np.ndarray, labels: np.ndarray, path: str) -> None:
    """PNG: grey image with cell borders in red."""
    from skimage.segmentation import find_boundaries
    rgb = np.stack([img_u8] * 3, axis=-1).copy()
    rgb[find_boundaries(labels, mode="inner")] = (255, 40, 40)
    Image.fromarray(rgb).save(path)


def list_images(folder: str) -> list:
    paths = []
    for e in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        paths += glob.glob(os.path.join(folder, e))
    return sorted(paths)


# =========================================================================
# micro_sam
# =========================================================================

# The package stores its recommended per-model thresholds under *_thresh keys,
# while InstanceSegmentationWithDecoder.generate() takes *_threshold arguments.
_AIS_KEY_MAP = {
    "center_distance_thresh": "center_distance_threshold",
    "boundary_distance_thresh": "boundary_distance_threshold",
}


def build_generate_kwargs(segmenter, model_type: str, overrides: dict) -> dict:
    """
    Keyword arguments for segmenter.generate():
      * the model's recommended center/boundary thresholds (what the napari
        plugin uses), if this segmenter is the decoder-based AIS one,
      * then any value the user set on the command line,
    keeping only arguments that this segmenter's generate() really accepts.
    """
    params = inspect.signature(segmenter.generate).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepted = set(params)
    kwargs = {}
    try:
        from micro_sam._model_settings import AIS_SETTINGS
        for src, dst in _AIS_KEY_MAP.items():
            if dst in accepted and src in AIS_SETTINGS.get(model_type, {}):
                kwargs[dst] = AIS_SETTINGS[model_type][src]
    except Exception:
        pass                                   # settings module not available: use generate() defaults
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in accepted and not takes_any:
            print(f"  note: '{k}' is not an argument of this segmentation mode and is ignored")
            continue
        kwargs[k] = v
    return kwargs


def load_micro_sam(model_type: str, device, mode):
    """Import micro_sam lazily and build the predictor + automatic segmenter."""
    try:
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    except ImportError as e:
        raise SystemExit(
            "micro_sam is not installed in this environment.\n"
            "Install it in a SEPARATE environment (see the micro_sam docs, "
            "https://computational-cell-analytics.github.io/micro-sam), e.g. on Colab or a new conda env.\n"
            f"(import error: {e})"
        )
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=None, device=device, segmentation_mode=mode,
    )
    return predictor, segmenter


def segment_cells(predictor, segmenter, img_u8: np.ndarray, gen_kwargs: dict) -> np.ndarray:
    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    return automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=img_u8, ndim=2, verbose=False, **gen_kwargs,
    )


# =========================================================================
# main
# =========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate cell label images with micro_sam.")
    ap.add_argument("--input", default="data/raw", help="folder with the raw images")
    ap.add_argument("--output", default="data/cells_microsam", help="where to write <stem>_cells.tif")
    ap.add_argument("--model", default="vit_b_lm", help="micro_sam model: vit_b_lm | vit_l_lm | vit_t_lm (light microscopy)")
    ap.add_argument("--mode", default="auto", choices=["auto", "ais", "amg", "apg"],
                    help="segmentation mode; 'auto' lets micro_sam choose (AIS when the model has a decoder)")
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: automatic)")
    ap.add_argument("--clip-percentiles", type=float, nargs=2, default=(0.5, 99.8), metavar=("LO", "HI"),
                    help="contrast stretch before segmentation; use 0 100 to disable")
    ap.add_argument("--min-cell-px", type=int, default=50, help="drop cells smaller than this many pixels (at input resolution)")
    # AIS tuning knobs (only passed when you set them)
    ap.add_argument("--center-distance-threshold", type=float, default=None)
    ap.add_argument("--boundary-distance-threshold", type=float, default=None)
    ap.add_argument("--foreground-threshold", type=float, default=None)
    ap.add_argument("--distance-smoothing", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true", help="redo images that already have a cells file")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images (testing)")
    args = ap.parse_args(argv)

    paths = list_images(args.input)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"No images found in {args.input}")
    os.makedirs(args.output, exist_ok=True)

    todo = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join(args.output, f"{stem}_cells.tif")) and not args.overwrite:
            print(f"skip {stem}: cells file exists (use --overwrite to redo)")
        else:
            todo.append(p)

    manifest_path = os.path.join(args.output, "cells_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    if todo:
        mode = None if args.mode == "auto" else args.mode
        print(f"Loading micro_sam model '{args.model}' (first use downloads the weights) ...")
        predictor, segmenter = load_micro_sam(args.model, args.device, mode)
        overrides = {
            "center_distance_threshold": args.center_distance_threshold,
            "boundary_distance_threshold": args.boundary_distance_threshold,
            "foreground_threshold": args.foreground_threshold,
            "distance_smoothing": args.distance_smoothing,
        }
        gen_kwargs = build_generate_kwargs(segmenter, args.model, overrides)
        print(f"Segmenter: {type(segmenter).__name__}   generate() arguments: {gen_kwargs}\n")

        try:
            import micro_sam
            version = getattr(micro_sam, "__version__", "unknown")
        except Exception:
            version = "unknown"
        manifest["_run"] = {
            "micro_sam_version": version, "model": args.model, "mode": args.mode,
            "segmenter": type(segmenter).__name__, "generate_kwargs": gen_kwargs,
            "clip_percentiles": list(args.clip_percentiles), "min_cell_px": args.min_cell_px,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for p in todo:
            stem = os.path.splitext(os.path.basename(p))[0]
            t0 = time.time()
            img_u8 = to_uint8(load_image(p), *args.clip_percentiles)
            inst = segment_cells(predictor, segmenter, img_u8, gen_kwargs)
            if inst is None:
                raise RuntimeError("micro_sam returned no segmentation")
            labels = clean_instances(np.asarray(inst), args.min_cell_px)
            stats = summarize(labels)
            save_cells(labels, os.path.join(args.output, f"{stem}_cells.tif"))
            save_preview(img_u8, labels, os.path.join(args.output, f"{stem}_cells_preview.png"))
            warns = sanity_warnings(stats)
            manifest[stem] = {**stats, "shape": list(labels.shape), "seconds": round(time.time() - t0, 1), "warnings": warns}
            print(f"{stem:<20s} cells={stats['n_cells']:4d}  covers {stats['foreground_percent']:5.1f}%  "
                  f"median area {stats['median_cell_area_px']:6d}px  ({time.time() - t0:.1f}s)")
            for w in warns:
                print(f"    WARNING: {w}")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone. Cell masks in {args.output}. Open the *_cells_preview.png files and check the red borders "
          f"follow the real cell outlines before setting CELL_SOURCE = 'microsam'.")


if __name__ == "__main__":
    main()