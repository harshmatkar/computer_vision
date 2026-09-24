"""
evaluate.py
-----------
Runs trained-model inference on raw .tif images for ONE of the three
registered Keras models, producing:
  1. Binary segmentation output masks (saved to outputs/masks_pred_<model>/)
  2. A per-image row in outputs/actin_metrics_<model>.csv with the four
     biophysical metrics (PER, MCC, RAT, SPI) plus Dice/F1/IoU against the
     classical+HOG label, inference time, and parameter count.

Usage:
    python evaluate.py --model unetpp
    python evaluate.py --model attention_unet
    python evaluate.py --model resunetpp
    python evaluate.py --all          # evaluate all three and print a
                                        # side-by-side summary for benchmarking
"""

import os
import time
import csv
import glob
import argparse

import numpy as np
import cv2
import tensorflow as tf

import config
from dataset import load_tif_image, preprocess_image
from model_builder import is_multi_output
from utils.metrics import compute_all_metrics


def dice_f1_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6):
    """
    Computes Dice coefficient, F1 score (identical to Dice for binary
    segmentation, included separately since benchmarking tables often
    report both under different names), and IoU (Jaccard index) between a
    predicted binary mask and a reference binary mask.
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


def load_model(model_name: str):
    """Loads the trained Keras model for `model_name` from its registered checkpoint."""
    registry_entry = config.MODEL_REGISTRY[model_name]
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, registry_entry["checkpoint_name"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run `python train.py --model {model_name}` first."
        )

    # custom_objects not needed: the saved .keras file embeds the full
    # architecture; compile=False since we only need inference here.
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    print(f"Loaded checkpoint from {checkpoint_path} ({model.count_params():,} parameters)")
    return model, model.count_params()


def run_inference(model, image_resized: np.ndarray, model_name: str):
    """Runs a single forward pass and returns the binary predicted mask plus inference time."""
    tensor = image_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    t0 = time.time()
    outputs = model(tensor, training=False)
    if is_multi_output(model_name):
        final_logits = outputs[-1]  # deepest / most-refined deep-supervision head
    else:
        final_logits = outputs
    probs = tf.sigmoid(final_logits)
    inference_time = time.time() - t0

    pred_mask = (probs.numpy().squeeze() >= config.SEGMENTATION_PROB_THRESHOLD).astype(np.uint8)
    return pred_mask, inference_time


def evaluate_model(model_name: str):
    """
    Main evaluation entry point for one registered model:
      - loads the trained model
      - iterates over every raw .tif image
      - runs the classical+HOG labeling pipeline (for the footprint mask and
        the reference label to score against)
      - runs model inference to get the predicted actin-edge mask
      - computes biophysical metrics + segmentation quality metrics
      - writes everything to outputs/actin_metrics_<model_name>.csv
      - saves predicted mask PNGs to outputs/masks_pred_<model_name>/
    """
    registry_entry = config.MODEL_REGISTRY[model_name]
    model, n_params = load_model(model_name)

    pred_mask_dir = os.path.join(config.OUTPUT_DIR, f"masks_pred_{model_name}")
    os.makedirs(pred_mask_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(config.RAW_IMAGE_DIR, "*.tif"))
        + glob.glob(os.path.join(config.RAW_IMAGE_DIR, "*.tiff"))
    )
    if len(image_paths) == 0:
        raise RuntimeError(f"No .tif/.tiff images found in {config.RAW_IMAGE_DIR}.")

    metrics_csv_path = os.path.join(config.OUTPUT_DIR, registry_entry["metrics_csv_name"])
    fieldnames = [
        "image_name", "model", "inference_time_sec", "model_num_parameters",
        "peripheral_enrichment_ratio", "membrane_coverage_continuity_pct",
        "radial_accumulation_thickness_px", "spatial_polarity_index_magnitude",
        "spatial_polarity_index_angle_deg", "dice_vs_label", "f1_vs_label",
        "iou_vs_label",
    ]

    all_rows = []
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in image_paths:
            name = os.path.basename(img_path)
            raw = load_tif_image(img_path)
            resized = cv2.resize(raw, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # The classical+HOG pipeline gives us the footprint mask (needed
            # for all four biophysical metrics) and the reference edge label
            # to score the model's prediction against.
            processed = preprocess_image(resized)
            footprint = processed["footprint"]
            reference_label = processed["edge_mask"]

            pred_mask, inference_time = run_inference(model, resized, model_name)

            seg_scores = dice_f1_iou(pred_mask, reference_label)
            biophysical = compute_all_metrics(
                resized, footprint, pred_mask,
                band_width_px=config.BAND_WIDTH_PX,
                num_sectors=config.POLARITY_NUM_SECTORS,
            )

            row = {
                "image_name": name,
                "model": model_name,
                "inference_time_sec": inference_time,
                "model_num_parameters": n_params,
                "dice_vs_label": seg_scores["dice"],
                "f1_vs_label": seg_scores["f1"],
                "iou_vs_label": seg_scores["iou"],
                **biophysical,
            }
            writer.writerow(row)
            all_rows.append(row)

            out_path = os.path.join(pred_mask_dir, os.path.splitext(name)[0] + "_pred.png")
            cv2.imwrite(out_path, pred_mask * 255)

            print(f"[{model_name}][{name}] dice={seg_scores['dice']:.3f} f1={seg_scores['f1']:.3f} "
                  f"iou={seg_scores['iou']:.3f} PER={biophysical['peripheral_enrichment_ratio']:.3f} "
                  f"MCC={biophysical['membrane_coverage_continuity_pct']:.1f}%")

    print(f"\nEvaluation complete for {model_name}. Metrics written to: {metrics_csv_path}")
    print(f"Predicted masks saved to: {pred_mask_dir}")
    return all_rows


def print_benchmark_summary(all_results: dict):
    """Prints a simple side-by-side mean-metric comparison across all evaluated models."""
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY (mean across all images)")
    print("=" * 78)
    header = f"{'Model':<18}{'Dice':>10}{'F1':>10}{'IoU':>10}{'Params':>14}{'Infer(s)':>12}"
    print(header)
    print("-" * 78)
    for model_name, rows in all_results.items():
        if not rows:
            continue
        dice = np.mean([r["dice_vs_label"] for r in rows])
        f1 = np.mean([r["f1_vs_label"] for r in rows])
        iou = np.mean([r["iou_vs_label"] for r in rows])
        params = rows[0]["model_num_parameters"]
        infer_t = np.mean([r["inference_time_sec"] for r in rows])
        print(f"{model_name:<18}{dice:>10.4f}{f1:>10.4f}{iou:>10.4f}{params:>14,}{infer_t:>12.4f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one or all of the three team-member models.")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                         choices=list(config.MODEL_REGISTRY.keys()),
                         help="Which model to evaluate.")
    parser.add_argument("--all", action="store_true",
                         help="Evaluate all three registered models and print a benchmark summary.")
    args = parser.parse_args()

    if args.all:
        results = {}
        for name in config.MODEL_REGISTRY.keys():
            try:
                results[name] = evaluate_model(name)
            except FileNotFoundError as e:
                print(f"Skipping {name}: {e}")
                results[name] = []
        print_benchmark_summary(results)
    else:
        evaluate_model(args.model)
