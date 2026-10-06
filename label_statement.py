"""
label_statement.py
------------------
Classical labeling pipeline that follows the project problem statement
step by step (Gaussian -> Otsu -> erosion boundary band -> targeted
intensity threshold -> binary mask, with HOG edge guidance).

This file is fully standalone: it does NOT import config.py, dataset.py or
anything else from the training code, so it cannot conflict with the
Frangi pipeline. All tunable values live in the `Params` dataclass below.

Problem statement  ->  function in this file
--------------------------------------------------------------------------
"Global Cell Segmentation"     smoothing filter + Otsu  -> build_footprint()
"Boundary Band Definition"     erode, subtract           -> boundary_band()
"HoG based labelling"          gradient-orientation      -> hog_edge_strength()
                               histograms (8x8-style cells)
"Targeted Intensity            higher threshold applied  -> label_image() step 5
 Thresholding"                 ONLY inside the band
"Binary Mask Extraction"       final cleaned binary mask -> label_image() step 7

Usage
-----
    # label every .tif in a folder, save masks + overlays + stage panels
    python label_statement.py --input data/raw --output outputs/labels_statement

    # or from Python / a notebook
    from label_statement import label_path, Params
    out = label_path("data/raw/1.tif")
    out["mask"]            # final binary actin mask (uint8, 0/1)
"""

import os
import glob
import argparse
from dataclasses import dataclass, replace, asdict

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


# =========================================================================
# PARAMETERS  (all sizes are in pixels AT image_size resolution)
# =========================================================================

@dataclass
class Params:
    # ---- working resolution -------------------------------------------
    image_size: tuple = (256, 256)

    # ---- 1. fine smoothing (noise removal; used for intensity tests) --
    fine_sigma: float = 1.0

    # ---- 2. global cell segmentation (footprint) ----------------------
    # Footprint is found on a heavier blur so fibre texture merges into a
    # solid "cellular footprint" (statement: "primary, solid binary mask").
    footprint_sigma: float = 3.0
    # Otsu is computed on sqrt-compressed intensities so the few very
    # bright actin pixels do not drag the threshold up into the cytoplasm
    # (set 1.0 for plain linear Otsu).
    footprint_gamma: float = 0.5
    # Multiplier on the Otsu value (1.0 = pure Otsu).
    footprint_otsu_scale: float = 1.0
    footprint_close_px: int = 5          # closing kernel diameter
    footprint_min_object_px: int = 400   # drop footprint specks smaller than this
    footprint_fill_hole_px: int = 300    # fill dark holes smaller than this
                                         # (nuclei / small voids); larger gaps
                                         # stay as background so their rims
                                         # count as periphery

    # ---- 3. boundary band (erosion) -----------------------------------
    band_radius_px: int = 8              # "predefined pixel radius"

    # ---- 4. targeted intensity threshold inside the band --------------
    # 'percentile' -> keep the top (100 - band_percentile)% of band pixels
    #                 (default; gives a continuous rim on dense monolayers)
    # 'otsu'       -> Otsu over the band pixels only (stricter: keeps only the
    #                 brightest rim segments, results can look fragmented)
    band_threshold_mode: str = "percentile"
    band_percentile: float = 55.0

    # ---- 5. HOG edge guidance -----------------------------------------
    use_hog: bool = True
    hog_cell_px: int = 4                 # HOG cell size (px)
    hog_bins: int = 9                    # unsigned orientation bins over 180 deg
    hog_percentile: float = 15.0         # keep band pixels whose HOG edge
                                         # strength is above this percentile
                                         # (computed within the band)

    # ---- 6. final clean-up --------------------------------------------
    final_close_px: int = 3              # bridge 1-2 px gaps along the rim
    final_min_object_px: int = 15        # remove isolated specks


# =========================================================================
# I/O
# =========================================================================

def load_tif_image(path: str) -> np.ndarray:
    """Load a (possibly 16-bit / multi-channel) image, min-max normalise to [0,1]."""
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.float32)


def list_images(folder: str):
    exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, e))
    return sorted(paths)


# =========================================================================
# STEP 2 - GLOBAL CELL SEGMENTATION
# =========================================================================

def _ellipse(k: int):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def build_footprint(img: np.ndarray, p: Params):
    """
    Gaussian smoothing -> Otsu threshold -> solid binary mask of the cells.

    Returns
    -------
    footprint : uint8 {0,1}
    coarse    : float32 heavily blurred image that Otsu was run on
    otsu_val  : float, the threshold actually used (in the gamma-compressed domain)
    """
    coarse = cv2.GaussianBlur(img, (0, 0), sigmaX=p.footprint_sigma)
    compressed = np.power(np.clip(coarse, 0, 1), p.footprint_gamma)

    otsu_val = float(threshold_otsu(compressed)) * p.footprint_otsu_scale
    fp = (compressed >= otsu_val).astype(np.uint8)

    # morphological clean-up -> "solid" footprint
    if p.footprint_close_px > 1:
        fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, _ellipse(p.footprint_close_px))

    # remove tiny foreground specks
    if fp.sum() > 0 and p.footprint_min_object_px > 0:
        fp = remove_small_objects(fp.astype(bool),
                                  max_size=p.footprint_min_object_px).astype(np.uint8)

    # fill small dark holes (nuclei / voids) but keep large gaps
    if p.footprint_fill_hole_px > 0:
        holes = ndi.binary_fill_holes(fp) & (fp == 0)
        lab, n = ndi.label(holes)
        if n > 0:
            areas = ndi.sum(holes, lab, index=np.arange(1, n + 1))
            small = np.isin(lab, 1 + np.where(areas <= p.footprint_fill_hole_px)[0])
            fp = np.where(small, 1, fp).astype(np.uint8)

    return fp.astype(np.uint8), coarse.astype(np.float32), otsu_val


# =========================================================================
# STEP 3 - BOUNDARY BAND
# =========================================================================

def boundary_band(footprint: np.ndarray, radius_px: int):
    """
    Erode the footprint by `radius_px` and subtract it from the original
    footprint -> a peripheral ring of width ~radius_px on the cell edge.

    Note: cv2.erode treats pixels outside the image as foreground, so the
    image frame itself does not create a fake band.
    """
    kernel = _ellipse(2 * int(radius_px) + 1)
    eroded = cv2.erode(footprint, kernel, iterations=1)
    band = (footprint.astype(bool) & ~eroded.astype(bool)).astype(np.uint8)
    return band, eroded


# =========================================================================
# STEP 4 - HOG EDGE-ORIENTATION STRENGTH
# =========================================================================

def hog_cell_histograms(img: np.ndarray, cell: int, bins: int):
    """
    Core of HOG (Dalal & Triggs, CVPR 2005) without block normalisation:
    per-cell histograms of gradient orientation (unsigned, 0-180 deg),
    weighted by gradient magnitude, with linear interpolation between the
    two nearest orientation bins.

    Block normalisation is deliberately skipped: it rescales every block to
    unit length and would erase the "how strong is this edge" information
    that we need for labeling.

    Returns hist of shape (H//cell, W//cell, bins).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)          # unsigned

    bin_w = 180.0 / bins
    pos = ang / bin_w - 0.5
    b0f = np.floor(pos)
    w1 = pos - b0f
    w0 = 1.0 - w1
    b0 = (b0f.astype(np.int32)) % bins
    b1 = (b0 + 1) % bins

    H, W = img.shape
    Hc, Wc = H // cell, W // cell
    hist = np.zeros((Hc, Wc, bins), dtype=np.float32)
    for b in range(bins):
        v = mag * (w0 * (b0 == b) + w1 * (b1 == b))
        v = v[:Hc * cell, :Wc * cell]
        hist[..., b] = v.reshape(Hc, cell, Wc, cell).sum(axis=(1, 3))
    return hist


def hog_edge_strength(img: np.ndarray, p: Params):
    """
    Per-pixel "oriented edge strength" map built from HOG cell histograms.

        strength(cell) = total gradient energy  x  orientation coherence

    where coherence = share of that energy in the dominant orientation
    (plus its two neighbouring bins). A cell crossing one clean edge has a
    single dominant orientation (high coherence); speckle noise spreads
    energy over all orientations (low coherence). The map is upsampled to
    image size and scaled to [0,1].
    """
    hist = hog_cell_histograms(img, p.hog_cell_px, p.hog_bins)
    energy = hist.sum(axis=-1)

    tri = hist + np.roll(hist, 1, axis=-1) + np.roll(hist, -1, axis=-1)
    coherence = tri.max(axis=-1) / (energy + 1e-6)

    strength_cells = energy * coherence
    strength = cv2.resize(strength_cells.astype(np.float32),
                          (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    # (cells that did not divide evenly leave a thin border; resize handles it)
    hi = np.percentile(strength, 99.5)
    if hi > 0:
        strength = np.clip(strength / hi, 0, 1)
    return strength.astype(np.float32)


# =========================================================================
# MAIN: LABEL ONE IMAGE
# =========================================================================

def label_image(img: np.ndarray, p: Params = None) -> dict:
    """
    Run the full pipeline on one normalised float image in [0,1].

    The image is resized to p.image_size first. Returns a dict with every
    intermediate stage (handy for the 'show the labeling steps' slide).
    """
    p = p or Params()
    h, w = p.image_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA
                     if img.shape[0] > h else cv2.INTER_LINEAR).astype(np.float32)

    # 1. fine smoothing
    smoothed = cv2.GaussianBlur(img, (0, 0), sigmaX=p.fine_sigma)

    # 2. global cell segmentation
    footprint, coarse, otsu_val = build_footprint(img, p)

    # 3. boundary band
    band, eroded = boundary_band(footprint, p.band_radius_px)

    # 4. HOG edge-orientation strength
    hog_strength = hog_edge_strength(smoothed, p) if p.use_hog else np.ones_like(smoothed)

    # 5. targeted intensity threshold INSIDE the band only
    band_bool = band.astype(bool)
    band_vals = smoothed[band_bool]
    if band_vals.size > 8:
        if p.band_threshold_mode == "otsu":
            band_thr = float(threshold_otsu(band_vals))
        else:
            band_thr = float(np.percentile(band_vals, p.band_percentile))
    else:
        band_thr = 1.0
    intensity_mask = (band_bool & (smoothed >= band_thr)).astype(np.uint8)

    # 6. HOG gating: keep candidates sitting on a coherent oriented edge
    if p.use_hog and band_vals.size > 8:
        hog_thr = float(np.percentile(hog_strength[band_bool], p.hog_percentile))
        mask = (intensity_mask.astype(bool) & (hog_strength >= hog_thr)).astype(np.uint8)
    else:
        hog_thr = 0.0
        mask = intensity_mask.copy()

    # 7. binary mask extraction (clean-up)
    if p.final_close_px > 1 and mask.sum() > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(p.final_close_px))
        mask = (mask.astype(bool) & band_bool).astype(np.uint8)   # stay inside band
    if mask.sum() > 0 and p.final_min_object_px > 0:
        mask = remove_small_objects(mask.astype(bool),
                                    max_size=p.final_min_object_px).astype(np.uint8)

    return {
        "image": img,
        "smoothed": smoothed,
        "coarse": coarse,
        "footprint": footprint,
        "eroded": eroded,
        "band": band,
        "hog_strength": hog_strength,
        "intensity_mask": intensity_mask,
        "mask": mask,
        "otsu_value": otsu_val,
        "band_threshold": band_thr,
        "hog_threshold": hog_thr,
        "actin_pixels": int(mask.sum()),
        "actin_percent": float(mask.sum()) / mask.size * 100.0,
        "params": asdict(p),
    }


def label_path(path: str, p: Params = None) -> dict:
    """Load an image file and label it."""
    out = label_image(load_tif_image(path), p)
    out["path"] = path
    return out


# =========================================================================
# VISUALISATION HELPERS
# =========================================================================

def make_overlay(img: np.ndarray, mask: np.ndarray, color=(0, 255, 80), alpha=0.55):
    """Grey image with the mask painted in `color` (RGB uint8 returned)."""
    rgb = np.stack([np.clip(img, 0, 1)] * 3, axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    layer = rgb.copy()
    layer[mask.astype(bool)] = color
    return cv2.addWeighted(rgb, 1 - alpha, layer, alpha, 0)


def build_panel_figure(out: dict, title: str = ""):
    """Build (but do not show/save) the stage-by-stage matplotlib figure."""
    import matplotlib.pyplot as plt

    boundary_overlay = make_overlay(out["image"], out["band"], color=(255, 60, 60), alpha=0.6)
    panels = [
        ("1. Original", out["image"], "gray"),
        ("2. Gaussian (fine)", out["smoothed"], "gray"),
        ("3. Otsu footprint", out["footprint"], "gray"),
        ("4. Boundary band", out["band"], "gray"),
        ("5. Band on image", boundary_overlay, None),
        ("6. HOG edge strength", out["hog_strength"], "magma"),
        ("7. Band intensity thr.", out["intensity_mask"], "gray"),
        ("8. Final actin mask", out["mask"], "gray"),
        ("9. Overlay", make_overlay(out["image"], out["mask"]), None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    for ax, (t, d, cm) in zip(axes, panels):
        ax.imshow(d, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}   actin px = {out['actin_pixels']} ({out['actin_percent']:.2f}%)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def save_panel(out: dict, save_path: str, title: str = ""):
    """Save the stage-by-stage figure to a PNG file."""
    import matplotlib.pyplot as plt
    fig = build_panel_figure(out, title)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# CLI
# =========================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")          # headless; notebooks keep their own backend
    ap = argparse.ArgumentParser(description="Problem-statement classical labeling.")
    ap.add_argument("--input", default="data/raw", help="folder with .tif images")
    ap.add_argument("--output", default="outputs/labels_statement", help="where to save results")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = list_images(args.input)
    if not paths:
        raise SystemExit(f"No images found in {args.input}")

    print(f"Labeling {len(paths)} image(s)  ->  {args.output}\n")
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        out = label_path(path)

        cv2.imwrite(os.path.join(args.output, f"{stem}_mask.png"), out["mask"] * 255)
        cv2.imwrite(os.path.join(args.output, f"{stem}_overlay.png"),
                    cv2.cvtColor(make_overlay(out["image"], out["mask"]), cv2.COLOR_RGB2BGR))
        save_panel(out, os.path.join(args.output, f"{stem}_panel.png"), title=stem)

        print(f"  {stem:<20s} actin px = {out['actin_pixels']:6d}  "
              f"({out['actin_percent']:.2f}%)   band px = {int(out['band'].sum())}")

    print("\nSaved <name>_mask.png, <name>_overlay.png, <name>_panel.png for each image.")


if __name__ == "__main__":
    main()
