"""
utils/metrics.py
------------------
Quantitative biophysical feature extraction for segmented actin-edge masks.

With the Frangi-based pipeline there is no separate cell-body footprint
(Otsu was removed). The `footprint` parameter passed here is a dilated
version of the actin mask itself (see dataset.preprocess_image), which
represents the "region near the membrane" for intensity comparisons.

Four metrics per image:
    PER  -- Peripheral Enrichment Ratio
    MCC  -- Membrane Coverage Continuity
    RAT  -- Radial Accumulation Thickness
    SPI  -- Spatial Polarity Index
"""

import numpy as np
import cv2


def peripheral_enrichment_ratio(image, footprint, edge_mask,
                                 band_width_px=6):
    """
    PER = mean_intensity(actin edge pixels) / mean_intensity(near-membrane
    pixels that are NOT part of the edge mask).

    With Frangi labeling: footprint = dilated actin mask (nearby region);
    "cytosolic" proxy = footprint pixels outside the edge mask.

    PER > 1 means actin is brighter at the detected ridges than in the
    surrounding membrane region (expected for true actin accumulation).
    """
    edge_bool = edge_mask.astype(bool)
    footprint_bool = footprint.astype(bool)
    nearby_bool = footprint_bool & (~edge_bool)

    edge_px   = image[edge_bool]
    nearby_px = image[nearby_bool]

    if edge_px.size == 0 or nearby_px.size == 0:
        return float("nan")

    mean_edge   = float(np.mean(edge_px))
    mean_nearby = float(np.mean(nearby_px))

    if mean_nearby == 0:
        return float("nan")
    return mean_edge / mean_nearby


def membrane_coverage_continuity(edge_mask, structuring_element_size=3):
    """
    MCC = connected length of actin ridges / total skeleton length * 100.

    Approximated by skeletonising the edge mask and measuring what
    fraction of skeleton pixels belong to connected components longer than
    a minimum length (> 5px), indicating continuous rather than fragmented
    actin coverage.
    """
    edge_u8 = (edge_mask > 0).astype(np.uint8)
    if edge_u8.sum() == 0:
        return float("nan")

    # Skeletonize to get 1-pixel-wide centerlines
    skeleton = cv2.ximgproc.thinning(edge_u8 * 255) // 255 \
        if hasattr(cv2, 'ximgproc') else edge_u8

    total_px = int(skeleton.sum())
    if total_px == 0:
        return float("nan")

    # Count pixels in connected components >= 5px (continuous stretches)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8)
    )
    continuous_px = sum(
        stats[i, cv2.CC_STAT_AREA]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 5
    )
    return float(continuous_px / total_px * 100)


def radial_accumulation_thickness(edge_mask):
    """
    RAT = average morphological width of the detected actin band.

    Estimated via the distance transform: for each foreground pixel, its
    value under the distance transform approximates the local half-width
    of the ridge at that point. Mean value * 2 gives average band width.
    """
    edge_u8 = (edge_mask > 0).astype(np.uint8)
    n_fg = int(edge_u8.sum())
    if n_fg == 0:
        return float("nan")
    if n_fg == edge_u8.size:
        return float("nan")

    dist = cv2.distanceTransform(edge_u8, cv2.DIST_L2, maskSize=5)
    vals = dist[dist > 0]
    if vals.size == 0:
        return float("nan")
    return float(2.0 * np.mean(vals))


def spatial_polarity_index(image, edge_mask, num_sectors=16):
    """
    SPI measures directional asymmetry of actin accumulation.

    The detected actin ridge pixels are projected around their centroid
    into `num_sectors` angular bins. The resultant vector magnitude
    (normalised by total intensity) is SPI_magnitude:
        0 = perfectly symmetric / uniform distribution
        1 = all actin in one direction (fully polarised)
    SPI_angle_degrees is the dominant direction.
    """
    ys, xs = np.nonzero(edge_mask.astype(bool))
    if ys.size == 0:
        return {"spi_magnitude": float("nan"), "spi_angle_degrees": float("nan")}

    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    intensities = image[ys, xs]
    angles = np.arctan2(ys - cy, xs - cx)

    sector_edges    = np.linspace(-np.pi, np.pi, num_sectors + 1)
    sector_mids     = (sector_edges[:-1] + sector_edges[1:]) / 2.0
    sector_sums     = np.zeros(num_sectors)
    sector_idx      = np.clip(np.digitize(angles, sector_edges) - 1, 0, num_sectors - 1)

    for s in range(num_sectors):
        sector_sums[s] = np.sum(intensities[sector_idx == s])

    total = sector_sums.sum()
    if total == 0:
        return {"spi_magnitude": float("nan"), "spi_angle_degrees": float("nan")}

    vx = np.sum(sector_sums * np.cos(sector_mids))
    vy = np.sum(sector_sums * np.sin(sector_mids))
    mag   = float(np.sqrt(vx**2 + vy**2) / total)
    angle = float(np.degrees(np.arctan2(vy, vx)) % 360)
    return {"spi_magnitude": mag, "spi_angle_degrees": angle}


def compute_all_metrics(image, footprint, edge_mask,
                         band_width_px=6, num_sectors=16):
    """
    Compute all four biophysical metrics and return as a flat dict.
    """
    per = peripheral_enrichment_ratio(image, footprint, edge_mask,
                                       band_width_px)
    mcc = membrane_coverage_continuity(edge_mask)
    rat = radial_accumulation_thickness(edge_mask)
    spi = spatial_polarity_index(image, edge_mask, num_sectors)

    return {
        "peripheral_enrichment_ratio":        per,
        "membrane_coverage_continuity_pct":   mcc,
        "radial_accumulation_thickness_px":   rat,
        "spatial_polarity_index_magnitude":   spi["spi_magnitude"],
        "spatial_polarity_index_angle_deg":   spi["spi_angle_degrees"],
    }
