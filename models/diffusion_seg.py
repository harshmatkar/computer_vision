"""
models/diffusion_seg.py
-----------------------
Conditional DIFFUSION segmenter (SegDiff-style) for the actin-edge masks.

Idea
----
The three other models are *discriminative*: image -> mask in one forward pass.
A diffusion model is *generative*: it learns to turn pure noise into a mask,
step by step, while looking at the image. Training:

    x0   = mask scaled to [-1, 1]                      (the clean "segmentation map")
    x_t  = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps        (add noise, t = 0..T-1, a_t = alpha-bar)
    net  : (image, x_t, t)  ->  eps_hat                (predict the noise that was added)
    loss : MSE(eps, eps_hat)                           <- the objective of this model

Sampling (DDIM, deterministic): start from x_T ~ N(0, I) and, for a short list
of steps, estimate x0 from the current x_t, then move to the next less-noisy x.
The result is random (it depends on the starting noise), so it is run
`n_samples` times and averaged -> a probability map (averaging several samples
helped a lot in testing: Dice 0.78 for one sample vs 0.92 for four, synthetic data).
The standard deviation across samples is returned too (`std`), but in testing
it was NOT higher on object pixels than on background, so treat it as
experimental until you have checked it on your own images.

What makes it different from Members A-C
  * network : small conditional U-Net with GroupNorm + time-embedding injection
              in every residual block (no BatchNorm, no skip-nesting, no SE/ASPP)
  * objective: noise-prediction MSE (denoising score matching), not Dice/Tversky/Combo
  * output   : a distribution of masks (mean + uncertainty), not a single mask

References
  * Amit, Nachmani, Shaharbany & Wolf. SegDiff: Image Segmentation with
    Diffusion Probabilistic Models. arXiv:2112.00390 (2021).
  * Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
  * Song, Meng & Ermon. Denoising Diffusion Implicit Models (DDIM). ICLR 2021.
  * Nichol & Dhariwal. Improved DDPM (cosine noise schedule). ICML 2021.
  (the last three are cited from memory - verify before putting them on a slide)

Training in this project goes through the normal `model.fit()` of train.py:
the dataset still yields (image, mask) pairs; the custom train_step below
turns each pair into (image, noisy mask, t, noise).
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# =========================================================================
# NOISE SCHEDULE + TIME EMBEDDING
# =========================================================================

def cosine_alpha_bars(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    Cumulative signal level a_t ("alpha-bar") for t = 0..T-1 using the cosine
    schedule of Nichol & Dhariwal. a_t starts near 1 (almost no noise) and
    decays to ~0 (almost pure noise) at t = T-1.
    """
    t = np.arange(timesteps + 1, dtype=np.float64) / timesteps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas).astype(np.float32)


def timestep_embedding(t, dim: int):
    """Sinusoidal embedding of integer time steps t [B] -> [B, dim] (Transformer-style)."""
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(half, dtype=tf.float32) / half)
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


# =========================================================================
# DENOISING NETWORK  (plain Keras functional model -> trivially saveable)
# =========================================================================

def _gn(x, name):
    ch = x.shape[-1]
    groups = 8 if ch % 8 == 0 else 1
    return layers.GroupNormalization(groups=groups, name=name)(x)


def _res_block(x, filters, t_emb, name):
    """GroupNorm-swish-conv, add a projection of the time embedding, GroupNorm-swish-conv, + skip."""
    h = _gn(x, f"{name}_gn1")
    h = layers.Activation("swish", name=f"{name}_act1")(h)
    h = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(h)

    tp = layers.Activation("swish", name=f"{name}_tact")(t_emb)
    tp = layers.Dense(filters, name=f"{name}_tproj")(tp)
    tp = layers.Reshape((1, 1, filters), name=f"{name}_treshape")(tp)
    h = layers.Add(name=f"{name}_tadd")([h, tp])

    h = _gn(h, f"{name}_gn2")
    h = layers.Activation("swish", name=f"{name}_act2")(h)
    h = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(h)

    skip = x if x.shape[-1] == filters else layers.Conv2D(filters, 1, name=f"{name}_skip")(x)
    return layers.Add(name=f"{name}_out")([h, skip])


def build_denoiser(image_size=(256, 256), base_filters=16, depth=3, time_dim=64) -> Model:
    """
    Conditional U-Net  eps_hat = f(image, x_t, t).

    inputs : image  (H, W, 1) in [-1, 1]   - the condition
             x_t    (H, W, 1)              - the current noisy mask
             t_emb  (time_dim,)            - sinusoidal embedding of the step
    output : (H, W, 1)                     - predicted noise (linear, no activation)
    H and W must be divisible by 2**depth.
    """
    h, w = image_size
    if h % (2 ** depth) or w % (2 ** depth):
        raise ValueError(f"image_size {image_size} must be divisible by 2**depth = {2 ** depth}")
    filters = [base_filters * (2 ** i) for i in range(depth + 1)]

    img = layers.Input((h, w, 1), name="image")
    xt = layers.Input((h, w, 1), name="noisy_mask")
    temb = layers.Input((time_dim,), name="t_embedding")

    t = layers.Dense(time_dim * 2, activation="swish", name="t_mlp1")(temb)
    t = layers.Dense(time_dim * 2, name="t_mlp2")(t)

    x = layers.Concatenate(name="cond_concat")([img, xt])
    x = layers.Conv2D(filters[0], 3, padding="same", name="stem")(x)

    skips = []
    for i in range(depth):
        x = _res_block(x, filters[i], t, f"down{i}_a")
        x = _res_block(x, filters[i], t, f"down{i}_b")
        skips.append(x)
        x = layers.Conv2D(filters[i], 3, strides=2, padding="same", name=f"down{i}_pool")(x)

    x = _res_block(x, filters[depth], t, "mid_a")
    x = _res_block(x, filters[depth], t, "mid_b")

    for i in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{i}_up")(x)
        x = layers.Conv2D(filters[i], 3, padding="same", name=f"up{i}_conv")(x)
        x = layers.Concatenate(name=f"up{i}_cat")([x, skips[i]])
        x = _res_block(x, filters[i], t, f"up{i}_a")
        x = _res_block(x, filters[i], t, f"up{i}_b")

    x = _gn(x, "out_gn")
    x = layers.Activation("swish", name="out_act")(x)
    out = layers.Conv2D(1, 1, kernel_initializer="zeros", name="eps_out")(x)
    return Model([img, xt, temb], out, name="Denoiser")


# =========================================================================
# THE DIFFUSION MODEL (training step + sampler)
# =========================================================================

class DiffusionSegmenter(Model):
    """
    Wraps a denoiser with the forward (noising) process, the training step and
    the DDIM sampler. Use it like any Keras model:  fit(train_ds, validation_data=val_ds).
    The dataset yields (image in [0,1], mask in {0,1}); both (B, H, W, 1).
    """

    def __init__(self, denoiser, timesteps=200, time_dim=64, prediction="eps", **kwargs):
        super().__init__(name="DiffusionSegmenter", **kwargs)
        if prediction not in ("eps", "x0"):
            raise ValueError("prediction must be 'eps' or 'x0'")
        self.denoiser = denoiser
        self.T = int(timesteps)
        self.time_dim = int(time_dim)
        self.prediction = prediction
        self.alpha_bars = tf.constant(cosine_alpha_bars(self.T))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.last_std = None            # uncertainty map of the most recent sample() call
        self.build(None)                # all weights live in the (already built) denoiser

    def build(self, input_shape=None):
        self.built = True

    def count_params(self):
        return int(self.denoiser.count_params())

    @property
    def metrics(self):
        return [self.loss_tracker]

    # ---- forward process --------------------------------------------------
    def _q_sample(self, x0, t, eps):
        ab = tf.gather(self.alpha_bars, t)[:, None, None, None]
        return tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * eps

    def _net(self, cond, x_t, t, training=False):
        return self.denoiser([cond, x_t, timestep_embedding(t, self.time_dim)], training=training)

    def _loss(self, data, t, eps, training):
        images, masks = data[0], data[1]
        cond = images * 2.0 - 1.0
        x0 = masks * 2.0 - 1.0
        x_t = self._q_sample(x0, t, eps)
        out = self._net(cond, x_t, t, training=training)
        target = eps if self.prediction == "eps" else x0
        return tf.reduce_mean(tf.square(target - out))

    # ---- training / validation steps (called by fit) -----------------------
    def train_step(self, data):
        b = tf.shape(data[0])[0]
        t = tf.random.uniform([b], 0, self.T, dtype=tf.int32)
        eps = tf.random.normal(tf.shape(data[1]))
        with tf.GradientTape() as tape:
            loss = self._loss(data, t, eps, training=True)
        vars_ = self.denoiser.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # fixed (stateless) randomness so val_loss is comparable between epochs
        b = tf.shape(data[0])[0]
        t = tf.random.stateless_uniform([b], seed=[11, 22], minval=0, maxval=self.T, dtype=tf.int32)
        eps = tf.random.stateless_normal(tf.shape(data[1]), seed=[33, 44])
        self.loss_tracker.update_state(self._loss(data, t, eps, training=False))
        return {"loss": self.loss_tracker.result()}

    # ---- reverse process -----------------------------------------------------
    def _eps(self, cond, x_t, t):
        """Predicted noise for integer step t (scalar). Separate method so tests can swap it."""
        tt = tf.fill([tf.shape(x_t)[0]], tf.cast(t, tf.int32))
        out = self._net(cond, x_t, tt, training=False)
        if self.prediction == "eps":
            return out
        ab = self.alpha_bars[t]                           # network predicted x0 -> convert to eps
        x0 = tf.clip_by_value(out, -1.0, 1.0)
        return (x_t - tf.sqrt(ab) * x0) / tf.sqrt(1.0 - ab)

    def sample(self, images, steps=20, n_samples=3, seed=0, return_trajectory=False):
        """
        DDIM sampling (eta = 0).

        images : float array (B, H, W, 1) in [0, 1]
        steps  : number of denoising steps actually run (<= T); fewer = faster
        n_samples : independent runs that are averaged (SegDiff merges several samples)

        Returns (prob, std): both (B, H, W) - mean mask probability and its std across samples.
        With return_trajectory=True also returns a list of the x0-estimates (in [0,1]) at
        every step of the FIRST sample of the FIRST image, for the "noise -> mask" figure.
        """
        images = tf.convert_to_tensor(images, tf.float32)
        cond = images * 2.0 - 1.0
        steps = int(min(max(steps, 1), self.T))
        ts = np.unique(np.linspace(self.T - 1, 0, steps).round().astype(np.int32))[::-1]
        gen = tf.random.Generator.from_seed(int(seed))
        ab_np = self.alpha_bars.numpy()

        samples, traj = [], []
        for s in range(int(n_samples)):
            x = gen.normal(tf.shape(images))
            for i, t in enumerate(ts):
                ab_t = float(ab_np[t])
                ab_prev = float(ab_np[ts[i + 1]]) if i + 1 < len(ts) else 1.0
                eps_hat = self._eps(cond, x, int(t))
                x0_hat = tf.clip_by_value((x - math.sqrt(1.0 - ab_t) * eps_hat) / math.sqrt(ab_t), -1.0, 1.0)
                eps_hat = (x - math.sqrt(ab_t) * x0_hat) / math.sqrt(1.0 - ab_t)   # consistent with the clipped x0
                x = math.sqrt(ab_prev) * x0_hat + math.sqrt(1.0 - ab_prev) * eps_hat
                if return_trajectory and s == 0:
                    traj.append(((x0_hat[0, :, :, 0].numpy() + 1.0) / 2.0))
            samples.append(((x[..., 0] + 1.0) / 2.0).numpy())
        stack = np.stack(samples, axis=0)
        prob, std = stack.mean(axis=0), stack.std(axis=0)
        self.last_std = std
        return (prob, std, traj) if return_trajectory else (prob, std)


# =========================================================================
# CALLBACK: Dice by actually sampling (val_loss alone says little about mask quality)
# =========================================================================

class DiffusionValDice(tf.keras.callbacks.Callback):
    """
    Every `every` epochs, sample masks for the validation set and write the Dice
    into logs["val_dice_metric"] (the same name train.py monitors for the other
    models). Between evaluations the last value is carried forward. Put this
    callback BEFORE ModelCheckpoint / EarlyStopping in the callback list.
    """

    def __init__(self, val_ds, every=5, steps=10, n_samples=1, threshold=0.5):
        super().__init__()
        self.val_ds, self.every, self.steps = val_ds, int(every), int(steps)
        self.n_samples, self.threshold = int(n_samples), float(threshold)
        self.last = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if (epoch + 1) % self.every == 0:
            dices = []
            for images, masks in self.val_ds:
                prob, _ = self.model.sample(images.numpy(), steps=self.steps, n_samples=self.n_samples, seed=epoch)
                pred = (prob >= self.threshold)
                gt = masks.numpy()[..., 0] > 0.5
                for p, g in zip(pred, gt):
                    dices.append((2.0 * (p & g).sum() + 1e-6) / (p.sum() + g.sum() + 1e-6))
            self.last = float(np.mean(dices))
        logs["val_dice_metric"] = self.last


# =========================================================================
# FACTORY used by train.py / evaluate.py
# =========================================================================

def build_diffusion(image_size=None, base_filters=None, depth=None, timesteps=None,
                    time_dim=None, prediction=None, learning_rate=None):
    """Build + compile the diffusion segmenter; unspecified values come from config.py."""
    import config
    image_size = image_size or tuple(config.IMAGE_SIZE)
    base_filters = base_filters or config.DIFFUSION_BASE_FILTERS
    depth = depth or config.DIFFUSION_DEPTH
    timesteps = timesteps or config.DIFFUSION_STEPS
    time_dim = time_dim or config.DIFFUSION_TIME_DIM
    prediction = prediction or config.DIFFUSION_PREDICTION
    lr = learning_rate or config.LEARNING_RATE

    denoiser = build_denoiser(image_size, base_filters, depth, time_dim)
    model = DiffusionSegmenter(denoiser, timesteps, time_dim, prediction)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=config.WEIGHT_DECAY, clipnorm=1.0),
        jit_compile=False,          # XLA is what broke GPU runs earlier; not needed here
    )
    return model


def count_parameters(model: Model) -> int:
    return int(model.count_params())
