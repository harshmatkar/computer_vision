# Actin Segmentation Pipeline — TensorFlow/Keras, 3-Model Benchmark

A complete pipeline for labeling and segmenting peripheral actin
accumulation in fluorescence microscopy images, built entirely in
**TensorFlow/Keras**. No manual annotation or Roboflow export is used —
labeling is done by a classical computer-vision pipeline (Gaussian → Otsu
→ morphological erosion → intensity threshold → HOG-guided refinement).
Three team members each train a distinct Keras architecture with a
distinct loss function against that same auto-generated label, for direct
benchmarking.

## 1. Project Structure

```
actin_segmentation_pipeline/
├── config.py                     # all hyperparameters, thresholds, model registry
├── dataset.py                    # labeling pipeline + tf.data.Dataset pipeline
├── model_builder.py               # maps a model name -> Keras architecture + loss
├── models/
│   ├── unet_plus_plus.py          # Member A: Lightweight UNet++ (SeparableConv2D)
│   ├── attention_unet.py          # Member B: Attention U-Net
│   └── resunet_pp.py              # Member C: ResUNet++ (Residual + SE + ASPP)
├── utils/
│   ├── hog_processing.py          # HOG features + boundary-band morphology
│   ├── losses.py                  # BCE+Dice, Tversky, Combo losses (one per member)
│   └── metrics.py                 # 4 biophysical metrics (PER, MCC, RAT, SPI)
├── train.py                       # training entry point, --model flag, Keras callbacks
├── evaluate.py                    # inference, mask export, per-model CSV, --all benchmark
├── data/
│   └── raw/                       # <-- put your .tif microscopy images here
├── checkpoints/                   # saved .keras model files, one per model
├── outputs/                       # predicted masks, CSVs, training history (per model)
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow automatically detects and uses a GPU if one is visible (no
manual device placement needed); `config.GPU_AVAILABLE` reports what was
detected at import time.

## 3. Add Your Data

Copy your microscopy images (`.tif` / `.tiff`) into:

```
data/raw/
```

**That's it — no annotation step required.** Every image is automatically
labeled by `dataset.preprocess_image()`:

1. **Global Cell Segmentation** — Gaussian smoothing, then Otsu's
   thresholding, to get a binary footprint mask of the whole cell.
2. **Boundary Band Definition** — morphological erosion of the footprint;
   subtracting the eroded mask from the original yields a thin ring
   strictly on the cell perimeter.
3. **Targeted Intensity Thresholding** — a secondary, higher-intensity
   percentile threshold applied only inside that boundary band isolates
   the highly-accumulated actin regions.
4. **Binary Mask Extraction** — binarize to produce the raw edge label.
5. **HOG-Guided Refinement** — the raw edge label is re-weighted by its
   Histogram-of-Oriented-Gradients response (`utils/hog_processing.py`);
   pixels that passed the intensity threshold by noise alone, but have no
   real gradient structure behind them, are suppressed before the final
   re-binarization.

`preprocess_image()` returns every intermediate stage (`smoothed`,
`footprint`, `boundary_band`, `edge_mask_raw`, `hog_weight_map`,
`edge_mask`) so you can show the raw image next to each processing stage
and the final label on your "Labeled Data" slide.

**Note on real fluorescence data:** if your actin signal appears as a
*darker* ring rather than a brighter one at the membrane (check your own
images first), the intensity-threshold direction in step 3 needs to be
flipped — see the `EDGE_INTENSITY_PERCENTILE` comment in `config.py`.

## 4. Train

Each team member trains their own model independently:

```bash
python train.py --model unetpp            # Member A: BCE + Dice loss
python train.py --model attention_unet    # Member B: Tversky loss
python train.py --model resunetpp         # Member C: Combo loss

# or train all three back to back:
python train.py --all
```

This uses the standard Keras `model.fit()` workflow with three callbacks:
- `ModelCheckpoint` — saves the best model (by validation Dice) to
  `checkpoints/best_model_<name>.keras`
- `CSVLogger` — logs per-epoch loss/Dice to `outputs/training_history_<name>.csv`
- `EarlyStopping` — stops training if validation Dice stalls, restoring
  the best-epoch weights

All hyperparameters (learning rate, batch size, epochs, per-loss weights,
image size, HOG refinement strength, etc.) live in `config.py`.

**Member A (UNet++) uses deep supervision** — a genuine multi-output Keras
model, with one loss/metric per decoder depth, averaged automatically by
`model.compile(loss=[...], loss_weights=[...])`. `train.py` handles
duplicating the target mask across the output heads automatically; you
don't need to do anything different when training this model versus the
single-output Members B and C.

## 5. Evaluate & Benchmark

```bash
python evaluate.py --model unetpp
python evaluate.py --model attention_unet
python evaluate.py --model resunetpp

# or evaluate all three and print a side-by-side comparison table:
python evaluate.py --all
```

Per model, this writes:
- Predicted binary masks to `outputs/masks_pred_<name>/`
- `outputs/actin_metrics_<name>.csv` with per-image biophysical metrics
  (Peripheral Enrichment Ratio, Membrane Coverage Continuity, Radial
  Accumulation Thickness, Spatial Polarity Index) plus Dice/F1/IoU against
  the classical+HOG label, inference time, and parameter count.

`--all` additionally prints a mean-metric comparison table across all
three architectures — the starting point for your box-plot benchmarking
figure.

## 6. The Three Models

| | Network | Loss Function | Reference Paper |
|---|---|---|---|
| **Member A** | Lightweight UNet++ (`SeparableConv2D`, ~285K params) | Combined BCE + Dice | Zhou et al., DLMIA 2018 |
| **Member B** | Attention U-Net (attention-gated skip connections) | Tversky Loss | Oktay et al., arXiv 2018 / Salehi et al., MLMI 2017 |
| **Member C** | ResUNet++ (Residual blocks + Squeeze-Excitation + ASPP) | Combo Loss (weighted-BCE + Dice) | Jha et al., IEEE ISM 2019 |

Each network uses a genuinely different core mechanism (nested skip depth
vs. spatial attention gating vs. channel-wise recalibration + multi-scale
atrous context), and each loss handles the foreground/background imbalance
of thin actin-edge masks differently — see the docstrings in
`utils/losses.py` and each file under `models/` for the full math and
reasoning behind each choice.

All three are built with the Keras **Functional API** (not `Sequential`
or subclassed `Model`), since every architecture here has branching,
merging, multi-input blocks (skip connections, attention gates, ASPP)
that the Functional API is designed for.

## 7. Extending the Pipeline

- Tune the classical labeling stage (thresholds, band width, HOG
  weighting) entirely from `config.py` without touching any other file.
- Add a fourth architecture by writing a new `build_<name>(...)` function
  under `models/` returning a Keras `Model` with input shape
  `(H, W, 1)` and output shape `(H, W, 1)` logits, then registering it in
  `config.MODEL_REGISTRY` and `model_builder.build_model()`.
- Add new biophysical metrics by writing a new function in
  `utils/metrics.py` and adding it to `compute_all_metrics()`.
