"""
evaluate.py
-----------
Runs trained-model inference on raw .tif images, produces:
  1. Binary segmentation output masks (saved as .png in config.OUTPUT_DIR/masks_pred/)
  2. A per-image row in actin_metrics.csv with the four biophysical metrics
     computed from utils/metrics.py (Peripheral Enrichment Ratio, Membrane
     Coverage Continuity, Radial Accumulation Thickness, Spatial Polarity
     Index).
  3. Standard segmentation quality metrics (Dice, F1, IoU) whenever a
     ground-truth mask is available, plus per-image inference time and
     model parameter count -- all needed for the benchmarking / box-plot
     section of the final report.
"""

import os
import time
import csv
import glob

import numpy as np
import cv2
import torch

import config
from dataset import load_tif_image, preprocess_image
from models.unet_plus_plus import LightUNetPlusPlus, count_parameters
from utils.metrics import compute_all_metrics


def dice_f1_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6):
    """
    Computes Dice coefficient, F1 score (identical to Dice for binary
    segmentation, included separately since some benchmarking tables
    report both under different names), and IoU (Jaccard index) between
    a predicted binary mask and a ground-truth binary mask.
    """
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {"dice": float(dice), "f1": float(f1), "iou": float(iou),
            "precision": float(precision), "recall": float(recall)}


def load_model(checkpoint_path: str = None) -> LightUNetPlusPlus:
    """Loads the trained model from a checkpoint saved by train.py."""
    checkpoint_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model_cfg = checkpoint.get("config", {})

    model = LightUNetPlusPlus(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_filters=model_cfg.get("base_filters", config.BASE_FILTERS),
        depth=model_cfg.get("depth", config.DEPTH),
        deep_supervision=model_cfg.get("deep_supervision", config.USE_DEEP_SUPERVISION),
    ).to(config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', '?')}, val_dice={checkpoint.get('val_dice', float('nan')):.4f})")
    return model


def run_inference(model: LightUNetPlusPlus, image_resized: np.ndarray):
    """
    Runs a single forward pass and returns the binary predicted mask plus
    inference time in seconds.
    """
    tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        final_logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        probs = torch.sigmoid(final_logits)
    inference_time = time.time() - t0

    pred_mask = (probs.squeeze().cpu().numpy() >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)
    return pred_mask, inference_time


def evaluate_all(checkpoint_path: str = None):
    """
    Main evaluation entry point:
      - loads the trained model
      - iterates over every raw .tif image
      - runs preprocessing (for footprint mask + optional GT comparison)
      - runs model inference to get the predicted actin-edge mask
      - computes biophysical metrics + segmentation quality metrics
      - writes everything to config.METRICS_CSV_PATH
      - saves predicted mask PNGs to OUTPUT_DIR/masks_pred/
    """
    model = load_model(checkpoint_path)
    n_params = count_parameters(model)

    pred_mask_dir = os.path.join(config.OUTPUT_DIR, "masks_pred")
    os.makedirs(pred_mask_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(config.RAW_IMAGE_DIR, "*.tif"))
        + glob.glob(os.path.join(config.RAW_IMAGE_DIR, "*.tiff"))
    )
    if len(image_paths) == 0:
        raise RuntimeError(f"No .tif/.tiff images found in {config.RAW_IMAGE_DIR}.")

    fieldnames = [
        "image_name", "inference_time_sec", "model_num_parameters",
        "peripheral_enrichment_ratio", "membrane_coverage_continuity_pct",
        "radial_accumulation_thickness_px", "spatial_polarity_index_magnitude",
        "spatial_polarity_index_angle_deg", "dice_vs_autolabel", "f1_vs_autolabel",
        "iou_vs_autolabel",
    ]

    with open(config.METRICS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in image_paths:
            name = os.path.basename(img_path)
            raw = load_tif_image(img_path)
            resized = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # Preprocessing pipeline gives us the footprint mask (needed for
            # all four biophysical metrics) and the weak/auto edge label
            # (used here as a reference to report Dice/F1/IoU against,
            # since it is available for every image without manual annotation).
            processed = preprocess_image(resized)
            footprint = processed["footprint"]
            auto_edge_label = processed["edge_mask"]

            pred_mask, inference_time = run_inference(model, resized)

            seg_scores = dice_f1_iou(pred_mask, auto_edge_label)
            biophysical = compute_all_metrics(
                resized, footprint, pred_mask,
                band_width_px=config.BAND_WIDTH_PX,
                num_sectors=config.POLARITY_NUM_SECTORS,
            )

            row = {
                "image_name": name,
                "inference_time_sec": inference_time,
                "model_num_parameters": n_params,
                "dice_vs_autolabel": seg_scores["dice"],
                "f1_vs_autolabel": seg_scores["f1"],
                "iou_vs_autolabel": seg_scores["iou"],
                **biophysical,
            }
            writer.writerow(row)

            # Save the predicted binary mask as a viewable PNG
            out_path = os.path.join(pred_mask_dir, os.path.splitext(name)[0] + "_pred.png")
            cv2.imwrite(out_path, pred_mask * 255)

            print(f"[{name}] dice={seg_scores['dice']:.3f} f1={seg_scores['f1']:.3f} "
                  f"iou={seg_scores['iou']:.3f} PER={biophysical['peripheral_enrichment_ratio']:.3f} "
                  f"MCC={biophysical['membrane_coverage_continuity_pct']:.1f}% "
                  f"RAT={biophysical['radial_accumulation_thickness_px']:.2f}px "
                  f"SPI={biophysical['spatial_polarity_index_magnitude']:.3f}")

    print(f"\nEvaluation complete. Metrics written to: {config.METRICS_CSV_PATH}")
    print(f"Predicted masks saved to: {pred_mask_dir}")


if __name__ == "__main__":
    evaluate_all()
