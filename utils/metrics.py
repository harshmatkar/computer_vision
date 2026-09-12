"""
utils/metrics.py
------------------
Quantitative biophysical feature extraction for segmented actin-edge masks.

Each function takes:
    - the original (smoothed) intensity image, float32 [0,1]
    - the binary cell footprint mask (whole cell), uint8 {0,1}
    - the binary edge/actin-accumulation mask (from the model or the
      preprocessing pipeline), uint8 {0,1}

and returns a single scalar (or vector, for polarity) biophysical metric.
All four metrics are combined by `compute_all_metrics()` into one dict per
cell, which `evaluate.py` then writes to `actin_metrics.csv`.
"""

import numpy as np
import cv2


def peripheral_enrichment_ratio(image: np.ndarray, footprint: np.ndarray,
                                 edge_mask: np.ndarray, band_width_px: int = 6) -> float:
    """
    Peripheral Enrichment Ratio (PER)
    ----------------------------------
    PER = mean_intensity(edge/peripheral actin) / mean_intensity(cytosolic actin)

    A PER > 1 indicates actin is preferentially accumulated at the cell
    periphery (e.g. cortical actin ring); PER ~ 1 indicates uniform
    cytoplasmic distribution.

    "Cytosolic actin" is defined here as all footprint pixels that are
    NOT part of the edge_mask and not part of the eroded boundary band,
    i.e. the deep interior of the cell.
    """
    footprint_bool = footprint.astype(bool)
    edge_bool = edge_mask.astype(bool)

    # Interior = footprint minus a `band_width_px`-wide erosion margin minus edge pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (band_width_px, band_width_px))
    eroded_interior = cv2.erode(footprint.astype(np.uint8), kernel, iterations=1).astype(bool)
    cytosolic_bool = eroded_interior & (~edge_bool)

    edge_pixels = image[edge_bool]
    cytosolic_pixels = image[cytosolic_bool]

    if edge_pixels.size == 0 or cytosolic_pixels.size == 0:
        return float("nan")

    mean_edge = float(np.mean(edge_pixels))
    mean_cytosolic = float(np.mean(cytosolic_pixels))

    if mean_cytosolic == 0:
        return float("nan")

    return mean_edge / mean_cytosolic


def membrane_coverage_continuity(footprint: np.ndarray, edge_mask: np.ndarray,
                                  structuring_element_size: int = 3) -> float:
    """
    Membrane Coverage Continuity (MCC)
    ------------------------------------
    MCC = (perimeter length actually covered by high-intensity edge actin)
          / (total perimeter length of the cell)  * 100%

    Implementation approach:
      1. Extract the 1-pixel-wide contour of the cell footprint via
         morphological gradient (dilation - erosion).
      2. Count contour pixels that overlap (or are adjacent to) edge_mask
         pixels as "covered".
      3. MCC = covered_contour_pixels / total_contour_pixels * 100.
    """
    footprint_u8 = (footprint > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (structuring_element_size, structuring_element_size))
    dilated = cv2.dilate(footprint_u8, kernel, iterations=1)
    eroded = cv2.erode(footprint_u8, kernel, iterations=1)
    contour_mask = (dilated - eroded).astype(bool)

    total_contour_px = int(np.sum(contour_mask))
    if total_contour_px == 0:
        return float("nan")

    # A contour pixel counts as "covered" if it or one of its 8-neighbors
    # is part of the edge mask (tolerance for slight misalignment).
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    covered_px = int(np.sum(contour_mask & edge_dilated))

    return (covered_px / total_contour_px) * 100.0


def radial_accumulation_thickness(footprint: np.ndarray, edge_mask: np.ndarray) -> float:
    """
    Radial Accumulation Thickness (RAT)
    --------------------------------------
    Average morphological width (in pixels) of the edge_mask band as it
    extends inward from the membrane, computed via the distance transform:
    for each edge-mask pixel, its distance-to-background value under a
    distance transform seeded from the *exterior* of the footprint
    approximates how many pixels deep that point is from the membrane.
    We instead directly measure band thickness using the Euclidean
    distance transform of the edge mask itself, which for a band-shaped
    region approximates half the local width at each point; doubling the
    mean of the ridge (skeleton) distance values gives the average band
    thickness.
    """
    edge_u8 = (edge_mask > 0).astype(np.uint8)
    n_fg = int(np.sum(edge_u8))
    if n_fg == 0:
        return float("nan")

    # Degenerate case: the "edge" mask covers the ENTIRE image, so there is
    # no background pixel anywhere to measure distance-to-boundary against.
    # OpenCV's distanceTransform then emits a float32 sentinel (~3.4e38) for
    # every pixel, which silently overflows on summation. This indicates a
    # failed/undertrained segmentation rather than a real thickness, so we
    # report NaN instead of a meaningless (or inf) number.
    if n_fg == edge_u8.size:
        return float("nan")

    # Distance transform: for each foreground pixel, distance to nearest
    # background (0) pixel. For a band of local half-width w, the maximum
    # distance transform value along the medial axis of the band is ~w.
    dist_transform = cv2.distanceTransform(edge_u8, cv2.DIST_L2, maskSize=5)

    # Estimate the local ridge (skeleton) via a simple non-maximum check:
    # pixels whose distance value is a local maximum along either axis are
    # treated as medial-axis samples of the band.
    ridge_values = dist_transform[dist_transform > 0]
    if ridge_values.size == 0:
        return float("nan")

    # Average thickness = 2x mean distance-to-background over foreground pixels,
    # since distance-to-background at a random band pixel averages ~ half-width.
    mean_thickness = 2.0 * float(np.mean(ridge_values))
    return mean_thickness


def spatial_polarity_index(image: np.ndarray, footprint: np.ndarray, edge_mask: np.ndarray,
                            num_sectors: int = 16):
    """
    Spatial Polarity Index (SPI)
    -------------------------------
    Quantifies directional asymmetry ("leading edge" polarization) of
    actin accumulation around the cell centroid.

    Method:
      1. Compute the cell centroid from the footprint mask.
      2. Divide the 360-degree angular range around the centroid into
         `num_sectors` equal sectors.
      3. Sum edge_mask-weighted intensity within each sector.
      4. Treat each sector's total as a vector of magnitude = sector sum
         and direction = sector's mid-angle; the resultant vector sum,
         normalized by the total sum across all sectors, gives:
            SPI_magnitude in [0, 1]  (0 = perfectly symmetric/uniform,
                                       1 = all actin concentrated in one
                                       direction -- fully polarized)
            SPI_angle_degrees        (dominant direction of polarization)

    Returns
    -------
    dict with keys: 'spi_magnitude' (float) and 'spi_angle_degrees' (float)
    """
    footprint_bool = footprint.astype(bool)
    ys, xs = np.nonzero(footprint_bool)
    if ys.size == 0:
        return {"spi_magnitude": float("nan"), "spi_angle_degrees": float("nan")}

    centroid_y, centroid_x = float(np.mean(ys)), float(np.mean(xs))

    edge_ys, edge_xs = np.nonzero(edge_mask.astype(bool))
    if edge_ys.size == 0:
        return {"spi_magnitude": float("nan"), "spi_angle_degrees": float("nan")}

    edge_intensities = image[edge_ys, edge_xs]

    # Angle (radians) of each edge pixel relative to the cell centroid
    angles = np.arctan2(edge_ys - centroid_y, edge_xs - centroid_x)

    sector_edges = np.linspace(-np.pi, np.pi, num_sectors + 1)
    sector_sums = np.zeros(num_sectors, dtype=np.float64)
    sector_mid_angles = (sector_edges[:-1] + sector_edges[1:]) / 2.0

    sector_idx = np.digitize(angles, sector_edges) - 1
    sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

    for s in range(num_sectors):
        mask_s = sector_idx == s
        sector_sums[s] = np.sum(edge_intensities[mask_s])

    total_sum = np.sum(sector_sums)
    if total_sum == 0:
        return {"spi_magnitude": float("nan"), "spi_angle_degrees": float("nan")}

    # Resultant vector: sum of (sector_sum * unit_vector(sector_mid_angle))
    vec_x = np.sum(sector_sums * np.cos(sector_mid_angles))
    vec_y = np.sum(sector_sums * np.sin(sector_mid_angles))
    resultant_magnitude = np.sqrt(vec_x ** 2 + vec_y ** 2)

    spi_magnitude = float(resultant_magnitude / total_sum)
    spi_angle_degrees = float(np.degrees(np.arctan2(vec_y, vec_x)) % 360)

    return {"spi_magnitude": spi_magnitude, "spi_angle_degrees": spi_angle_degrees}


def compute_all_metrics(image: np.ndarray, footprint: np.ndarray, edge_mask: np.ndarray,
                         band_width_px: int = 6, num_sectors: int = 16) -> dict:
    """
    Convenience wrapper: computes all four biophysical metrics for one
    cell/image and returns them as a single flat dictionary, ready to be
    appended as a row to actin_metrics.csv.
    """
    per_value = peripheral_enrichment_ratio(image, footprint, edge_mask, band_width_px)
    mcc_value = membrane_coverage_continuity(footprint, edge_mask)
    rat_value = radial_accumulation_thickness(footprint, edge_mask)
    spi = spatial_polarity_index(image, footprint, edge_mask, num_sectors)

    return {
        "peripheral_enrichment_ratio": per_value,
        "membrane_coverage_continuity_pct": mcc_value,
        "radial_accumulation_thickness_px": rat_value,
        "spatial_polarity_index_magnitude": spi["spi_magnitude"],
        "spatial_polarity_index_angle_deg": spi["spi_angle_degrees"],
    }
