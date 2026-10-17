"""
label_statement.py  (v2 - per-cell boundary bands)
--------------------------------------------------
Classical labeling pipeline that follows the project problem statement
step by step, applied PER CELL so that it also finds the actin between
neighbouring cells in a dense monolayer.

Problem statement                       ->  where it happens here
----------------------------------------------------------------------------
Global Cell Segmentation                    build_footprint()
  Gaussian smoothing + Otsu threshold        (Otsu, with hysteresis so dim
  -> solid binary mask of the cells           cells and thin protrusions stay)
(extension) split the mask into cells       segment_cells()
  marker-controlled watershed                 each cell gets its own boundary
Boundary Band Definition                    boundary_bands()
  erode each cell by a pixel radius,          = "cell minus eroded cell", done
  subtract -> peripheral ring                 for every cell
HoG based labelling                         hog_edge_strength()
  gradient-orientation histograms             oriented-edge strength map
Targeted Intensity Thresholding             label_image(), steps 5-6
  higher threshold ONLY inside the bands
Binary Mask Extraction                      label_image(), step 7

Why per cell?
  A single footprint for the whole image has only ONE outer edge, so on a
  dense monolayer the band never reaches the cell-cell junctions that
  contain most of the actin. Giving every cell its own footprint and band
  puts the junctions inside bands, which is exactly the problem statement's
  idea ("cell boundary vs internal cytoplasm") applied cell by cell.

Small, standard additions to the statement's method (cite in the report):
  * hysteresis thresholding for the footprint        Canny (1986)
  * marker-controlled watershed to split cells        Meyer & Beucher (1990s)
  * white top-hat + local (Weber-type) contrast       classical morphology
  * HOG-style orientation histograms                  Dalal & Triggs (2005)
  * Otsu thresholding                                 Otsu (1979)

Cells can also come from OUTSIDE (e.g. a foundation model such as micro_sam,
Archit et al., Nature Methods 2025): pass `cells=<label image>` to
label_image()/label_path() and steps 1-2 above (Otsu footprint + watershed)
are skipped; the per-cell bands, HOG and intensity steps run unchanged on
the supplied cells. See prepare_external_cells() for how they are cleaned.

This file is standalone: it does not import config.py or dataset.py.

Usage
-----
    python label_statement.py --input data/raw --output outputs/labels_statement

    from label_statement import label_path, Params
    out = label_path("data/raw/1.tif")
    out["mask"]        # final binary actin mask (uint8, 0/1)
"""

import os
import glob
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import h_minima, remove_small_objects
from skimage.segmentation import watershed, find_boundaries, expand_labels


# =========================================================================
# PARAMETERS  (pixel sizes are AT image_size resolution, default 256x256)
# =========================================================================

@dataclass
class Params:
    image_size: tuple = (256, 256)
    fine_sigma: float = 1.0                  # noise-removal Gaussian

    # ---- 1. global cell segmentation (footprint) ---------------------
    footprint_sigma: float = 2.0             # blur before Otsu
    footprint_gamma: float = 0.5             # sqrt-compress so a few very bright pixels
                                             # do not pull Otsu up (1.0 = linear)
    footprint_otsu_scale: float = 1.0        # multiplier on the Otsu value
    footprint_low_frac: float = 0.8          # hysteresis: pixels above 0.8*Otsu count as
                                             # cell if connected to pixels above Otsu
                                             # (lower = keep dimmer cells, but the mask
                                             # starts to bleed into dark background)
    footprint_close_px: int = 3
    footprint_min_object_px: int = 150       # drop footprint specks smaller than this
    footprint_fill_hole_px: int = 200        # fill dark holes smaller than this

    # ---- 2. split into cells (marker-controlled watershed) -----------
    cell_marker_sigma: float = 3.0           # blur used to find dark cell interiors
    cell_marker_h: float = 0.012             # depth of an interior minimum to count as a
                                             # cell (lower = more, smaller cells)
    cell_elev_sigma: float = 1.5

    # ---- 3. boundary bands (erode each cell, subtract) ---------------
    band_radius_inner_px: int = 3            # between two neighbouring cells
    band_radius_outer_px: int = 8            # along the edge facing background

    # ---- 4. contrast measure used for the intensity threshold --------
    contrast_tophat_px: int = 9              # white top-hat: keeps bright structures
                                             # narrower than this
    contrast_norm_sigma: float = 10.0        # local brightness scale
    contrast_norm_eps: float = 0.10

    # ---- 5. HOG ------------------------------------------------------
    use_hog: bool = True
    hog_cell_px: int = 4
    hog_bins: int = 9

    # ---- 6. targeted thresholds inside the bands (percentiles taken
    #         over band pixels only) ------------------------------------
    strong_contrast_percentile: float = 70.0
    strong_hog_percentile: float = 60.0
    weak_contrast_percentile: float = 55.0
    weak_hog_percentile: float = 45.0
    reach_px: int = 3                        # weak pixels are kept only within this
                                             # distance of a strong pixel (bridges gaps
                                             # along a junction, cannot flood the band)

    # ---- 7. clean-up -------------------------------------------------
    final_close_px: int = 3
    final_min_object_px: int = 12

    # ---- external cells (only used when cells=... is passed) ---------
    external_cells_min_px: int = 30          # drop cell labels smaller than this (at image_size)
    external_cells_expand_px: int = 2        # grow every cell by this many px WITHOUT overlap, so the
                                             # thin gap that segmenters often leave between neighbouring
                                             # cells (where the junction actin lies) falls inside a band


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
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return arr.astype(np.float32)


def list_images(folder: str):
    paths = []
    for e in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        paths += glob.glob(os.path.join(folder, e))
    return sorted(paths)


def _ellipse(k: int):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# =========================================================================
# STEP 1 - GLOBAL CELL SEGMENTATION
# =========================================================================

def build_footprint(img: np.ndarray, p: Params):
    """
    Gaussian smoothing -> Otsu threshold (with hysteresis) -> solid binary
    mask of everything that is cell.

    Returns footprint (uint8 0/1), otsu_value (in the gamma-compressed domain).
    """
    coarse = cv2.GaussianBlur(img, (0, 0), sigmaX=p.footprint_sigma)
    comp = np.power(np.clip(coarse, 0, 1), p.footprint_gamma)

    t_high = float(threshold_otsu(comp)) * p.footprint_otsu_scale
    t_low = t_high * p.footprint_low_frac
    fp = apply_hysteresis_threshold(comp, t_low, t_high).astype(np.uint8)

    if p.footprint_close_px > 1:
        fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, _ellipse(p.footprint_close_px))

    if fp.sum() > 0 and p.footprint_min_object_px > 0:
        fp = remove_small_objects(fp.astype(bool),
                                  max_size=p.footprint_min_object_px).astype(np.uint8)

    if p.footprint_fill_hole_px > 0:
        holes = ndi.binary_fill_holes(fp) & (fp == 0)
        lab, n = ndi.label(holes)
        if n > 0:
            areas = ndi.sum(holes, lab, index=np.arange(1, n + 1))
            small = np.isin(lab, 1 + np.where(areas <= p.footprint_fill_hole_px)[0])
            fp = np.where(small, 1, fp).astype(np.uint8)

    return fp, t_high


# =========================================================================
# STEP 2 - SPLIT THE FOOTPRINT INTO INDIVIDUAL CELLS
# =========================================================================

def segment_cells(img: np.ndarray, footprint: np.ndarray, p: Params):
    """
    Marker-controlled watershed. Dark cell interiors are the markers; the
    flooding stops on the bright ridges, so the cell borders (watershed
    lines) follow the actin-rich junctions. Footprint pieces that received
    no marker (isolated cells) become one cell each.

    Returns an int label image (0 = background, 1..N = cells) and N.
    """
    fpb = footprint.astype(bool)
    if not fpb.any():
        return np.zeros(footprint.shape, np.int32), 0

    interior = cv2.GaussianBlur(img, (0, 0), sigmaX=p.cell_marker_sigma)
    markers, n = ndi.label(h_minima(interior, p.cell_marker_h) & fpb)

    comp, nc = ndi.label(fpb)
    has_marker = np.zeros(nc + 1, bool)
    has_marker[np.unique(comp[markers > 0])] = True
    nxt = n + 1
    for c in range(1, nc + 1):
        if not has_marker[c]:
            markers[comp == c] = nxt
            nxt += 1

    elevation = cv2.GaussianBlur(img, (0, 0), sigmaX=p.cell_elev_sigma)
    labels = watershed(elevation, markers, mask=fpb)
    return labels.astype(np.int32), int(nxt - 1)


# =========================================================================
# EXTERNAL CELLS (e.g. from micro_sam)
# =========================================================================

def load_cells_file(path: str) -> np.ndarray:
    """
    Load an integer cell-label image (0 = background, 1..N = cells) saved as
    .tif/.tiff/.png (any integer dtype) or .npy. Returns a 2-D int array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(path)
        except ImportError:                       # tifffile ships with scikit-image, but be safe
            with Image.open(path) as im:
                arr = np.array(im)
    else:
        with Image.open(path) as im:
            arr = np.array(im)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Cell label image must be 2-D, got shape {arr.shape} from {path}")
    return arr


def _resize_labels_nearest(lab: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize of a label image using pixel-centre mapping, so label
    boundaries stay aligned with an intensity image resized by cv2.INTER_AREA/LINEAR."""
    H, W = lab.shape
    ys = np.minimum(((np.arange(h) + 0.5) * H / h).astype(np.int64), H - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * W / w).astype(np.int64), W - 1)
    return lab[np.ix_(ys, xs)]


def prepare_external_cells(cells, image_size, p: Params) -> np.ndarray:
    """
    Turn an externally produced cell-label image into the label image the
    band step expects:
      1. resize to image_size with nearest-neighbour (labels are never blended)
      2. make ids compact (1..N), 0 stays background
      3. drop cells smaller than p.external_cells_min_px
      4. grow all cells by p.external_cells_expand_px without letting them
         overlap (skimage.segmentation.expand_labels), closing the thin gaps
         that segmenters tend to leave between neighbours
    Returns int32 labels of shape image_size.
    """
    lab = np.asarray(cells)
    lab = np.squeeze(lab)
    if lab.ndim != 2:
        raise ValueError(f"cells must be a 2-D label image, got shape {lab.shape}")
    if lab.dtype.kind == "f":
        lab = np.rint(lab)
    lab = np.clip(lab, 0, None).astype(np.int64)

    h, w = image_size
    if lab.shape != (h, w):
        lab = _resize_labels_nearest(lab, h, w)

    uniq, inv = np.unique(lab, return_inverse=True)       # compact ids, safe for huge label values
    lab = inv.reshape(lab.shape).astype(np.int64)
    if uniq[0] != 0:                                      # no background pixel present
        lab = lab + 1

    if p.external_cells_min_px > 0 and lab.max() > 0:
        counts = np.bincount(lab.ravel())
        small = counts < p.external_cells_min_px
        small[0] = False
        lab[small[lab]] = 0

    if p.external_cells_expand_px > 0 and lab.max() > 0:
        lab = expand_labels(lab, distance=p.external_cells_expand_px)

    uniq, inv = np.unique(lab, return_inverse=True)       # re-compact after removals
    lab = inv.reshape(lab.shape)
    if uniq[0] != 0:
        lab = lab + 1
    return lab.astype(np.int32)


# =========================================================================
# STEP 3 - BOUNDARY BANDS
# =========================================================================

def boundary_bands(labels: np.ndarray, r_inner: int, r_outer: int):
    """
    For every cell: erode it by a radius and subtract -> ring along its
    edge. Pixels within r of the cell edge form the ring, so the union of
    all cells' rings is computed exactly (and fast) with distance
    transforms of the cell-edge pixels.

    Two radii: r_outer along edges that face background (the real cell
    edge, which the blurred footprint tends to place slightly outside the
    bright rim) and r_inner along edges shared by two neighbouring cells
    (where the watershed line already sits on the junction ridge).
    The image frame itself is not treated as a cell edge.

    Returns band (uint8 0/1) and the two edge maps for display.
    """
    L = labels.astype(np.int32)
    H, W = L.shape
    pad = np.pad(L, 1, mode="edge")
    outer_edge = np.zeros((H, W), bool)
    inner_edge = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        outer_edge |= (L > 0) & (nb == 0)
        inner_edge |= (L > 0) & (nb > 0) & (nb != L)

    d_out = ndi.distance_transform_edt(~outer_edge)
    d_in = ndi.distance_transform_edt(~inner_edge)
    band = (L > 0) & ((d_out <= r_outer) | (d_in <= r_inner))
    return band.astype(np.uint8), outer_edge, inner_edge


# =========================================================================
# STEP 4 - HOG EDGE-ORIENTATION STRENGTH
# =========================================================================

def hog_cell_histograms(img: np.ndarray, cell: int, bins: int):
    """
    Core of HOG (Dalal & Triggs, CVPR 2005) without block normalisation:
    per-cell histograms of gradient orientation (unsigned, 0-180 deg),
    weighted by gradient magnitude, linearly interpolated between the two
    nearest bins. Block normalisation is skipped on purpose: it rescales
    every block to unit length and would erase how strong an edge is.
    Returns (H//cell, W//cell, bins).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0

    pos = ang / (180.0 / bins) - 0.5
    b0f = np.floor(pos)
    w1 = pos - b0f
    w0 = 1.0 - w1
    b0 = b0f.astype(np.int32) % bins
    b1 = (b0 + 1) % bins

    H, W = img.shape
    Hc, Wc = H // cell, W // cell
    hist = np.zeros((Hc, Wc, bins), np.float32)
    for b in range(bins):
        v = (mag * (w0 * (b0 == b) + w1 * (b1 == b)))[:Hc * cell, :Wc * cell]
        hist[..., b] = v.reshape(Hc, cell, Wc, cell).sum(axis=(1, 3))
    return hist


def hog_edge_strength(img: np.ndarray, p: Params):
    """
    Per-pixel oriented-edge strength from HOG cell histograms:
        strength = total gradient energy x orientation coherence
    (coherence = share of the energy in the dominant orientation and its
    two neighbours). A clean edge has one dominant orientation; speckle
    spreads its energy over all orientations. Upsampled, scaled to [0,1].
    """
    hist = hog_cell_histograms(img, p.hog_cell_px, p.hog_bins)
    energy = hist.sum(axis=-1)
    tri = hist + np.roll(hist, 1, axis=-1) + np.roll(hist, -1, axis=-1)
    coherence = tri.max(axis=-1) / (energy + 1e-6)

    s = cv2.resize((energy * coherence).astype(np.float32),
                   (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    hi = np.percentile(s, 99.5)
    return (np.clip(s / hi, 0, 1) if hi > 0 else s).astype(np.float32)


# =========================================================================
# MAIN: LABEL ONE IMAGE
# =========================================================================

def label_image(img: np.ndarray, p: Params = None, cells=None) -> dict:
    """
    Run the full pipeline on one normalised float image in [0,1]
    (it is resized to p.image_size first). Returns a dict with every
    intermediate stage plus the final binary mask under "mask".

    cells : optional. A 2-D integer cell-label image (any size) or a path to
            one (.tif/.png/.npy). If given, Otsu footprint + watershed are
            skipped and these cells are used instead (see
            prepare_external_cells). out["cell_source"] tells which was used.
    """
    p = p or Params()
    h, w = p.image_size
    interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR
    img = cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)

    smoothed = cv2.GaussianBlur(img, (0, 0), sigmaX=p.fine_sigma)

    # 1-2. footprint, then individual cells
    if cells is None:
        footprint, otsu_val = build_footprint(img, p)
        cell_labels, n_cells = segment_cells(img, footprint, p)
        cell_source = "classical"
    else:
        if isinstance(cells, (str, os.PathLike)):
            cells = load_cells_file(str(cells))
        cell_labels = prepare_external_cells(cells, (h, w), p)
        n_cells = int(cell_labels.max())
        footprint = (cell_labels > 0).astype(np.uint8)
        otsu_val = float("nan")
        cell_source = "external"

    # 3. per-cell boundary bands
    band, outer_edge, inner_edge = boundary_bands(
        cell_labels, p.band_radius_inner_px, p.band_radius_outer_px)
    bb = band.astype(bool)

    empty = {
        "image": img, "smoothed": smoothed, "footprint": footprint, "cells": cell_labels,
        "n_cells": n_cells, "cell_source": cell_source, "band": band,
        "outer_edge": outer_edge, "inner_edge": inner_edge,
        "contrast": np.zeros_like(img), "hog_strength": np.zeros_like(img),
        "strong": np.zeros_like(band), "mask": np.zeros_like(band),
        "otsu_value": otsu_val, "actin_pixels": 0, "actin_percent": 0.0, "params": asdict(p),
    }
    if bb.sum() < 16:
        return empty

    # 4. contrast measure: white top-hat (thin bright structures), divided by
    #    local brightness so dim and bright regions are judged fairly
    tophat = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, _ellipse(p.contrast_tophat_px))
    local = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=p.contrast_norm_sigma)
    contrast = tophat / (local + p.contrast_norm_eps)

    # 5. HOG oriented-edge strength
    hog = hog_edge_strength(smoothed, p) if p.use_hog else np.ones_like(img)

    # 6. targeted thresholds INSIDE the bands (percentiles over band pixels)
    def pct(x, q):
        return float(np.percentile(x[bb], q))

    strong = bb & (contrast >= pct(contrast, p.strong_contrast_percentile))
    weak = bb & (contrast >= pct(contrast, p.weak_contrast_percentile))
    if p.use_hog:
        strong &= hog >= pct(hog, p.strong_hog_percentile)
        weak &= hog >= pct(hog, p.weak_hog_percentile)

    # weak pixels survive only if connected to a strong pixel AND within reach_px of one
    near = cv2.dilate(strong.astype(np.uint8), _ellipse(2 * p.reach_px + 1)).astype(bool)
    lab_w, n_w = ndi.label(weak & near, structure=np.ones((3, 3)))
    keep = np.zeros(n_w + 1, bool)
    keep[np.unique(lab_w[strong])] = True
    keep[0] = False
    mask = (keep[lab_w] | strong).astype(np.uint8)

    # 7. binary mask extraction
    if p.final_close_px > 1 and mask.sum() > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(p.final_close_px))
        mask &= band
    if mask.sum() > 0 and p.final_min_object_px > 0:
        mask = remove_small_objects(mask.astype(bool),
                                    max_size=p.final_min_object_px).astype(np.uint8)

    return {
        "image": img, "smoothed": smoothed, "footprint": footprint, "cells": cell_labels,
        "n_cells": n_cells, "cell_source": cell_source, "band": band,
        "outer_edge": outer_edge, "inner_edge": inner_edge,
        "contrast": contrast.astype(np.float32), "hog_strength": hog,
        "strong": strong.astype(np.uint8), "mask": mask,
        "otsu_value": otsu_val,
        "actin_pixels": int(mask.sum()),
        "actin_percent": float(mask.sum()) / mask.size * 100.0,
        "params": asdict(p),
    }


def label_path(path: str, p: Params = None, cells=None) -> dict:
    """Load an image file and label it. `cells`: see label_image()."""
    out = label_image(load_tif_image(path), p, cells=cells)
    out["path"] = path
    return out


# =========================================================================
# VISUALISATION HELPERS
# =========================================================================

def make_overlay(img: np.ndarray, mask: np.ndarray, color=(0, 255, 80), alpha=0.55):
    """Grey image with the mask painted in `color` (RGB uint8 returned)."""
    rgb = (np.stack([np.clip(img, 0, 1)] * 3, axis=-1) * 255).astype(np.uint8)
    layer = rgb.copy()
    layer[mask.astype(bool)] = color
    return cv2.addWeighted(rgb, 1 - alpha, layer, alpha, 0)


def cell_outline_overlay(img: np.ndarray, cells: np.ndarray):
    """Grey image with the cell borders drawn in red."""
    rgb = (np.stack([np.clip(img, 0, 1)] * 3, axis=-1) * 255).astype(np.uint8)
    rgb[find_boundaries(cells, mode="inner")] = (255, 50, 50)
    return rgb


def build_panel_figure(out: dict, title: str = ""):
    """Stage-by-stage matplotlib figure (2 rows x 5 columns). Not shown/saved here."""
    import matplotlib.pyplot as plt

    panels = [
        ("1. Original", out["image"], "gray"),
        ("2. Gaussian smoothing", out["smoothed"], "gray"),
        ("3. Otsu footprint", out["footprint"], "gray"),
        (f"4. Cells (watershed, n={out['n_cells']})", cell_outline_overlay(out["image"], out["cells"]), None),
        ("5. Boundary bands (red)", make_overlay(out["image"], out["band"], (255, 60, 60), 0.6), None),
        ("6. Local contrast (top-hat)", np.clip(out["contrast"] / max(np.percentile(out["contrast"], 99.5), 1e-6), 0, 1), "magma"),
        ("7. HOG edge strength", out["hog_strength"], "magma"),
        ("8. Strong pixels in band", out["strong"], "gray"),
        ("9. Final actin mask", out["mask"], "gray"),
        ("10. Overlay", make_overlay(out["image"], out["mask"]), None),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    for ax, (t, d, cm) in zip(axes.ravel(), panels):
        ax.imshow(d, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}   actin px = {out['actin_pixels']} ({out['actin_percent']:.2f}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.18)
    return fig


def save_panel(out: dict, save_path: str, title: str = ""):
    import matplotlib.pyplot as plt
    fig = build_panel_figure(out, title)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# CLI
# =========================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")          # headless; notebooks keep their own backend
    ap = argparse.ArgumentParser(description="Problem-statement classical labeling (v2).")
    ap.add_argument("--input", default="data/raw", help="folder with images")
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
        print(f"  {stem:<20s} cells = {out['n_cells']:4d}   actin px = {out['actin_pixels']:6d} "
              f"({out['actin_percent']:.2f}%)")
    print("\nSaved <name>_mask.png, <name>_overlay.png, <name>_panel.png for each image.")


if __name__ == "__main__":
    main()