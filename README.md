# Actin Segmentation Pipeline — 3-Model Benchmark

A complete pipeline for labeling and segmenting peripheral actin
accumulation in fluorescence microscopy images. **No manual annotation or
Roboflow export is used** — labeling is done entirely by a classical
computer-vision pipeline (Gaussian → Otsu → morphological erosion →
intensity threshold → HOG-guided refinement). Three team members each
train a distinct deep learning architecture with a distinct loss function
against that same auto-generated label, for direct benchmarking.

## 1. Project Structure

```
actin_segmentation_pipeline/
├── config.py                     # all hyperparameters, thresholds, model registry
├── dataset.py                    # labeling pipeline + PyTorch Dataset/DataLoader
├── model_builder.py               # maps a model name -> architecture + loss function
├── models/
│   ├── unet_plus_plus.py          # Member A: Lightweight UNet++ (depthwise separable convs)
│   ├── attention_unet.py          # Member B: Attention U-Net
│   └── resunet_pp.py              # Member C: ResUNet++ (Residual + SE + ASPP)
├── utils/
│   ├── hog_processing.py          # HOG features + boundary-band morphology
│   ├── losses.py                  # BCE+Dice, Tversky, Combo losses (one per member)
│   └── metrics.py                 # 4 biophysical metrics (PER, MCC, RAT, SPI)
├── train.py                       # training loop, --model flag, checkpointing
├── evaluate.py                    # inference, mask export, per-model CSV, --all benchmark
├── data/
│   └── raw/                       # <-- put your .tif microscopy images here
├── checkpoints/                   # saved model weights, one file per model
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

## 3. Add Your Data

Copy your microscopy images (`.tif` / `.tiff`) into:

```
data/raw/
```

**That's it — no annotation step required.** Every image is automatically
labeled by `dataset.preprocess_image()`, which implements the exact
pipeline your project brief specifies:

1. **Global Cell Segmentation** — Gaussian smoothing, then Otsu's
   thresholding, to get a binary footprint mask of the whole cell.
2. **Boundary Band Definition** — morphological erosion of the footprint;
   subtracting the eroded mask from the original yields a thin ring
   strictly on the cell perimeter.
3. **Targeted Intensity Thresholding** — a secondary, higher-intensity
   percentile threshold applied only inside that boundary band isolates
   the highly-accumulated actin regions.
4. **Binary Mask Extraction** — binarize to produce the raw edge label.
5. **HOG-Guided Refinement** — the raw edge label from step 4 is re-weighted
   by its Histogram-of-Oriented-Gradients response (`utils/hog_processing.py`).
   HOG's gradient-orientation histograms are strongest exactly at sharp
   intensity transitions (true membrane edges), so pixels that passed the
   intensity threshold by noise alone — but have no real gradient structure
   behind them — are suppressed. Only pixels that are both intensity-bright
   **and** gradient-consistent with a genuine boundary survive into the
   final label.

For your "Presentation of Labeled Data" slide, `preprocess_image()` returns
every intermediate stage (`smoothed`, `footprint`, `boundary_band`,
`edge_mask_raw`, `hog_weight_map`, `edge_mask`) so you can show the raw
image next to each processing stage and the final label side by side.

## 4. Train

Each team member trains their own model independently:

```bash
python train.py --model unetpp            # Member A: BCE + Dice loss
python train.py --model attention_unet    # Member B: Tversky loss
python train.py --model resunetpp         # Member C: Combo loss

# or train all three back to back:
python train.py --all
```

This will, per model:
- Split your data into train/validation sets (`config.VAL_SPLIT`)
- Train with that model's registered loss function (see `config.MODEL_REGISTRY`)
- Save the best checkpoint to `checkpoints/best_model_<name>.pth`
- Log per-epoch loss/Dice to `outputs/training_history_<name>.csv`

All hyperparameters (learning rate, batch size, epochs, per-loss weights,
image size, HOG refinement strength, etc.) live in `config.py`.

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

`--all` additionally prints a mean-metric comparison table across all three
architectures — the starting point for your box-plot benchmarking figure.

## 6. The Three Models

| | Network | Loss Function | Reference Paper |
|---|---|---|---|
| **Member A** | Lightweight UNet++ (depthwise separable convs, ~287K params) | Combined BCE + Dice | Zhou et al., DLMIA 2018 |
| **Member B** | Attention U-Net (attention-gated skip connections) | Tversky Loss | Oktay et al., arXiv 2018 / Salehi et al., MLMI 2017 |
| **Member C** | ResUNet++ (Residual blocks + Squeeze-Excitation + ASPP) | Combo Loss (weighted-BCE + Dice) | Jha et al., IEEE ISM 2019 |

Each network uses a genuinely different core mechanism (nested skip depth
vs. spatial attention gating vs. channel-wise recalibration + multi-scale
atrous context), and each loss handles the foreground/background imbalance
of thin actin-edge masks differently — see the docstrings in
`utils/losses.py` and each file under `models/` for the full math and
reasoning behind each choice.

## 7. Extending the Pipeline

- Tune the classical labeling stage (thresholds, band width, HOG weighting)
  entirely from `config.py` without touching any other file.
- Add a fourth architecture by writing a new file under `models/` with a
  `forward(x) -> logits` (or list of logits, for deep supervision)
  interface, then registering it in `config.MODEL_REGISTRY` and
  `model_builder.build_model()`.
- Add new biophysical metrics by writing a new function in
  `utils/metrics.py` and adding it to `compute_all_metrics()`.
