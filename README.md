# Actin Segmentation Pipeline

A complete pipeline for segmenting and quantifying peripheral actin
accumulation in fluorescence microscopy images, using classical image
processing for weak-label generation plus a lightweight UNet++ (depthwise
separable convolutions) for learned segmentation.

## 1. Project Structure

```
actin_segmentation_pipeline/
├── config.py                    # all hyperparameters, thresholds, paths
├── dataset.py                   # preprocessing + PyTorch Dataset/DataLoader
├── models/
│   └── unet_plus_plus.py        # lightweight UNet++ (depthwise separable convs)
├── utils/
│   ├── hog_processing.py        # HOG features + boundary-band morphology
│   └── metrics.py                # 4 biophysical metrics (PER, MCC, RAT, SPI)
├── train.py                      # training loop, checkpointing, history CSV
├── evaluate.py                   # inference, mask export, actin_metrics.csv
├── data/
│   ├── raw/                      # <-- put your .tif microscopy images here
│   └── masks/                    # (optional) hand-labeled ground-truth masks
├── checkpoints/                  # saved model weights (created automatically)
├── outputs/                      # predicted masks, CSVs, training history
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

If you have hand-labeled ground-truth masks, put them in `data/masks/` with
the **same base filename** as their corresponding image (e.g.
`cell_003.tif` -> `cell_003.png`). If no manual mask exists for an image,
the pipeline automatically generates a weak label using the classical
Gaussian -> Otsu -> erosion -> intensity-threshold chain described in
`dataset.py`.

## 4. Train

```bash
python train.py
```

This will:
- Split your data into train/validation sets (`config.VAL_SPLIT`)
- Train the lightweight UNet++ with a combined BCE + Dice loss across all
  deep-supervision output depths
- Save the best checkpoint (highest validation Dice) to
  `checkpoints/best_model.pth`
- Log per-epoch loss/Dice to `outputs/training_history.csv`

All hyperparameters (learning rate, batch size, epochs, loss weights, image
size, etc.) can be changed in `config.py` without touching any other file.

## 5. Evaluate

```bash
python evaluate.py
```

This will:
- Load `checkpoints/best_model.pth`
- Run inference on every image in `data/raw/`
- Save predicted binary masks to `outputs/masks_pred/`
- Write per-image biophysical metrics (Peripheral Enrichment Ratio,
  Membrane Coverage Continuity, Radial Accumulation Thickness, Spatial
  Polarity Index) plus segmentation quality metrics (Dice, F1, IoU),
  inference time, and model parameter count to `outputs/actin_metrics.csv`

## 6. Key Design Notes

- **Depthwise Separable Convolutions**: every conv block in the UNet++
  uses a depthwise 3x3 conv followed by a pointwise 1x1 conv instead of a
  standard 3x3 conv, cutting parameter count/FLOPs by roughly 8-9x versus a
  standard-convolution UNet++ of the same width — this is what makes the
  network "lightweight" enough to train quickly on modest hardware.
- **UNet++ nested skip connections**: intermediate convolutional nodes
  progressively bridge the semantic gap between encoder and decoder
  features, which is especially useful for thin, boundary-hugging
  structures like the actin edge band.
- **Weak-label bootstrapping**: when no expert annotation exists, the
  classical Gaussian/Otsu/erosion/intensity-threshold chain in
  `dataset.py` supplies an automatically generated label so the model can
  still be trained; this is standard practice in bio-image pipelines with
  limited manual annotation budgets.
- **HOG-guided boundary refinement**: `utils/hog_processing.py` provides
  an optional soft-weighting step that uses HOG gradient-orientation
  strength to down-weight likely-spurious boundary-band pixels before they
  become training labels.

## 7. Extending the Pipeline

- Swap in your own labeling tool's masks by dropping PNGs into
  `data/masks/` — no code changes required.
- Add new biophysical metrics by writing a new function in
  `utils/metrics.py` and adding it to `compute_all_metrics()`.
- To benchmark against other architectures, add a new file under
  `models/` following the same `forward(x) -> logits` interface used by
  `LightUNetPlusPlus`, then swap the import in `train.py`/`evaluate.py`.
