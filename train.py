"""
train.py
--------
Training entry point shared by all three team-member models, using the
standard Keras `model.fit()` workflow. Select which model to train with
the --model flag:

    python train.py --model unetpp             # Member A: BCE + Dice loss
    python train.py --model attention_unet      # Member B: Tversky loss
    python train.py --model resunetpp             # Member C: Combo loss
    python train.py --all                         # train all three sequentially

Each model gets its own checkpoint file (.keras) and training-history CSV
(see config.MODEL_REGISTRY), so all three can be trained independently and
then benchmarked side by side in evaluate.py / the final report.

Implementation notes:
  - Loss function is selected per model in model_builder.build_and_compile()
  - Member A (UNet++) uses deep supervision: a multi-output Keras model.
    Since tf.data yields a single (image, mask) pair per example, this
    file duplicates the mask once per output head via model_builder's
    num_outputs() before calling model.fit() -- Keras requires the target
    structure to match the model's output structure for multi-output models.
  - Checkpointing, CSV history logging, and early stopping are all handled
    by standard Keras callbacks (ModelCheckpoint, CSVLogger, EarlyStopping).
"""

import os
import time
import argparse

import tensorflow as tf

import config
from dataset import get_datasets
from model_builder import build_and_compile, is_multi_output, num_outputs


def _duplicate_target_for_multi_output(ds: tf.data.Dataset, n_outputs: int) -> tf.data.Dataset:
    """
    Reshapes a (image, mask) tf.data.Dataset into (image, (mask, mask, ...))
    with `n_outputs` copies of the mask, matching a multi-output model's
    expected target structure (used only for UNet++'s deep supervision).
    """
    return ds.map(lambda img, mask: (img, tuple(mask for _ in range(n_outputs))),
                  num_parallel_calls=tf.data.AUTOTUNE)


def train_model(model_name: str = None):
    """
    Main training entry point for one of the three registered models.
    """
    model_name = model_name or config.DEFAULT_MODEL
    if model_name not in config.MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(config.MODEL_REGISTRY.keys())}")

    registry_entry = config.MODEL_REGISTRY[model_name]

    print(f"=== Training {registry_entry['display_name']} (model_name='{model_name}') ===")
    print(f"GPU available: {config.GPU_AVAILABLE}")
    print(f"Loss function: {registry_entry['loss']}")

    train_ds, val_ds, n_train, n_val = get_datasets()
    print(f"Train images: {n_train} | Val images: {n_val}")

    model = build_and_compile(model_name)
    n_params = model.count_params()
    print(f"Model parameter count: {n_params:,}")

    if is_multi_output(model_name):
        n_out = num_outputs(model_name)
        train_ds = _duplicate_target_for_multi_output(train_ds, n_out)
        val_ds = _duplicate_target_for_multi_output(val_ds, n_out)

    history_path = os.path.join(config.OUTPUT_DIR, f"training_history_{model_name}.csv")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, registry_entry["checkpoint_name"])

    # Keras tracks a separate metric per output for multi-output models,
    # named "val_<output_layer_name>_<metric_fn_name>". For UNet++ we
    # monitor the deepest output (the one actually used at inference time).
    if is_multi_output(model_name):
        monitor_metric = f"val_output_{config.DEPTH}_dice_metric"
    else:
        monitor_metric = "val_dice_metric"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor=monitor_metric, mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric, mode="max",
            patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1,
        ),
    ]

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.NUM_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    total_time = time.time() - t0

    print(f"Training complete for {model_name} in {total_time:.1f}s.")
    print(f"Best checkpoint saved to: {ckpt_path}")
    print(f"Training history saved to: {history_path}")
    return model, history_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one of the three team-member segmentation models.")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                         choices=list(config.MODEL_REGISTRY.keys()),
                         help="Which model to train: unetpp (Member A), attention_unet (Member B), "
                              "resunetpp (Member C).")
    parser.add_argument("--all", action="store_true",
                         help="Train all three registered models sequentially.")
    args = parser.parse_args()

    if args.all:
        for name in config.MODEL_REGISTRY.keys():
            train_model(name)
    else:
        train_model(args.model)
