"""
train.py
--------
Training entry point for all three registered models.

    python train.py --model unetpp          # Member A: Focal + Dice
    python train.py --model attention_unet  # Member B: Focal Tversky
    python train.py --model resunetpp       # Member C: Weighted-CE + Dice
    python train.py --all                   # train all three sequentially

CHANGELOG:
    - Added ReduceLROnPlateau callback.
      With the new augmented dataset (50 steps/epoch) and lower starting
      LR (5e-4), ReduceLROnPlateau halves the learning rate automatically
      when val dice does not improve for 7 consecutive epochs, preventing
      oscillation around a local minimum in later training.
      This is a standard best-practice addition; no other training logic
      changed.
"""

import os
import time
import argparse

import tensorflow as tf

import config
from dataset import get_datasets
from model_builder import build_and_compile, is_multi_output, num_outputs


def _duplicate_target_for_multi_output(ds: tf.data.Dataset,
                                        n_outputs: int) -> tf.data.Dataset:
    """
    Reshape (image, mask) -> (image, (mask, mask, ...)) with n_outputs
    copies of the mask to match UNet++'s deep-supervision output structure.
    Applied AFTER augmentation so the duplicated masks are the augmented ones.
    """
    return ds.map(
        lambda img, mask: (img, tuple(mask for _ in range(n_outputs))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def train_model(model_name: str = None):
    """Train one registered model end-to-end."""
    model_name    = model_name or config.DEFAULT_MODEL
    registry_entry = config.MODEL_REGISTRY[model_name]

    print(f"=== Training {registry_entry['display_name']} ===")
    print(f"GPU available   : {config.GPU_AVAILABLE}")
    print(f"Loss function   : {registry_entry['loss']}")
    print(f"LR              : {config.LEARNING_RATE}")
    print(f"Augment repeats : {config.AUGMENT_REPEATS}")

    train_ds, val_ds, n_train, n_val = get_datasets()
    print(f"Train images    : {n_train}  |  Val images: {n_val}")

    # Estimate steps per epoch for logging
    steps_per_epoch = (n_train * config.AUGMENT_REPEATS) // config.BATCH_SIZE
    print(f"Steps/epoch     : {steps_per_epoch}  "
          f"(was ~{n_train // config.BATCH_SIZE} before augmentation)")

    model  = build_and_compile(model_name)
    n_params = model.count_params()
    print(f"Parameters      : {n_params:,}")

    if is_multi_output(model_name):
        n_out    = num_outputs(model_name)
        train_ds = _duplicate_target_for_multi_output(train_ds, n_out)
        val_ds   = _duplicate_target_for_multi_output(val_ds,   n_out)

    history_path = os.path.join(
        config.OUTPUT_DIR, f"training_history_{model_name}.csv"
    )
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR, registry_entry["checkpoint_name"]
    )

    # Metric to monitor -- the deepest head for multi-output models
    if is_multi_output(model_name):
        monitor_metric = f"val_output_{config.DEPTH}_dice_metric"
    else:
        monitor_metric = "val_dice_metric"

    callbacks = [
        # Save best checkpoint by val dice
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        # Log full training history to CSV
        tf.keras.callbacks.CSVLogger(history_path),

        # Stop if val dice does not improve for EARLY_STOP_PATIENCE epochs
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        # ── NEW: ReduceLROnPlateau ────────────────────────────────────
        # Halve LR when val dice fails to improve for 7 epochs.
        # Prevents oscillating around a local minimum in later training
        # and allows continued learning even after the initial LR is
        # exhausted. min_lr floors it at 1e-6 to avoid numerical issues.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="max",
            factor=0.5,       # multiply LR by 0.5 on plateau
            patience=7,       # wait 7 epochs before reducing
            min_lr=1e-6,
            verbose=1,
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

    print(f"\nTraining complete in {total_time:.1f}s.")
    print(f"Best checkpoint : {ckpt_path}")
    print(f"History CSV     : {history_path}")
    return model, history_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=config.DEFAULT_MODEL,
        choices=list(config.MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all three models sequentially.",
    )
    args = parser.parse_args()

    if args.all:
        for name in config.MODEL_REGISTRY.keys():
            train_model(name)
    else:
        train_model(args.model)
