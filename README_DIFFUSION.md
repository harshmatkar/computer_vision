# Diffusion segmenter (add-on branch)

A fourth model, `diffusion`, next to `unetpp`, `attention_unet` and `resunetpp`:
a **conditional diffusion model** (SegDiff-style) that *generates* the actin mask from noise while looking at the image.

| | Members A-C | Diffusion |
|---|---|---|
| Output | one mask in one forward pass | many sampled masks -> their mean |
| Network | U-Net variants, BatchNorm | small conditional U-Net, GroupNorm, time embedding in every residual block |
| Objective | BCE+Dice / Tversky / Combo | noise-prediction MSE (denoising score matching) |

## Files in this zip (unzip into the project root)
| File | Status |
|---|---|
| `models/diffusion_seg.py` | NEW - network, forward process, training step, DDIM sampler, validation callback |
| `config.py` | CHANGED - `"diffusion"` registry entry + `DIFFUSION_*` block (everything else exactly as you had it) |
| `train.py` | CHANGED - `--model diffusion` (own train step, weights-only checkpoint, sampled-Dice monitor) |
| `evaluate.py` | CHANGED - diffusion loading + sampling + uncertainty PNGs |
| `diffusion_samples.ipynb` | NEW - samples, std map, noise->mask trajectory, effect of #samples |
| `changes_diffusion.diff` | exact diff of the three changed files vs the versions you pasted |

Not included on purpose: `model_builder.py`, `models/*.py` (other three), `utils/*`, `dataset.py` - I have not seen your current versions
and overwriting them could undo your changes. Nothing in them needs to change for diffusion.
To see/apply only my edits to the three changed files: `patch -p1 --dry-run < changes_diffusion.diff` (drop `--dry-run` to apply).

## Run
```bash
python train.py --model diffusion          # trains, saves checkpoints/best_model_diffusion.weights.h5
python evaluate.py --model diffusion       # samples masks, metrics CSV, outputs/uncertainty_diffusion/*.png
jupyter notebook diffusion_samples.ipynb
```
`evaluate.py --all` now includes it (skipped if not trained). `loocv.py` / `kfold_eval.py` do NOT support it (a diffusion fold = training + sampling; not wired).

## How it works (what to explain in the report)
Mask x0 is scaled to [-1,1]. Training picks a random step t, adds noise: `x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*eps`
(`a_t` = cosine schedule, T=`DIFFUSION_STEPS`), and the network `f(image, x_t, t)` must predict `eps`. Loss = `mean((eps - eps_hat)^2)`.
Sampling (DDIM, deterministic): start from noise; at each of `DIFFUSION_SAMPLING_STEPS` steps estimate
`x0_hat = (x_t - sqrt(1-a_t)*eps_hat)/sqrt(a_t)` (clipped to [-1,1]) and move to the next, less noisy `x_t`.
Because the start noise is random, `DIFFUSION_NUM_SAMPLES` masks are generated and averaged; threshold at `SEGMENTATION_PROB_THRESHOLD`.
Checkpoint selection uses a *sampled* validation Dice (every `DIFFUSION_VAL_EVERY` epochs) because noise-MSE alone says little about mask quality.

## Knobs (`config.py`)
`DIFFUSION_BASE_FILTERS`, `DIFFUSION_DEPTH` (IMAGE_SIZE must be divisible by 2**depth), `DIFFUSION_STEPS` (T),
`DIFFUSION_PREDICTION` (`"eps"` or `"x0"`), `DIFFUSION_SAMPLING_STEPS`, `DIFFUSION_NUM_SAMPLES`, `DIFFUSION_VAL_EVERY`, `DIFFUSION_VAL_STEPS`.

## Cost and memory - read before running
* Inference cost per image = `SAMPLING_STEPS x NUM_SAMPLES` network passes (default 20 x 4 = 80). Much slower than Members A-C.
* It uses `config.IMAGE_SIZE`. Your pasted config has **512x512**: memory grows with pixel count, so on a ~1.7 GiB GPU use 256x256, batch 1, or train on CPU (`CUDA_VISIBLE_DEVICES=-1`).
* The model compiles with `jit_compile=False` (XLA is not used).

## What was and was not verified
Verified (sandbox, CPU, synthetic data):
* sampler math: with the true noise supplied it reconstructs the mask exactly (5, 20 and 200 steps); a useless denoiser does not
* it learns: Dice on 8 unseen synthetic images 0.78 with one sample, 0.92 with four averaged; saved weights reload to identical samples
* patched `train.py` / `evaluate.py` run end to end; the original model path through `train.py` is unchanged
NOT verified:
* anything on your real images, GPU behaviour, or speed on your machine
* that it beats the other three. On the synthetic task a plain threshold scored 0.93 vs 0.92 for the diffusion model - so do not assume it is better.
* the std map: it was NOT higher on object pixels than background in the test - treat as experimental.

## References
* Amit, Nachmani, Shaharbany, Wolf. SegDiff: Image Segmentation with Diffusion Probabilistic Models. arXiv:2112.00390 (2021) - verified.
* Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020 - from memory, verify.
* Song, Meng, Ermon. Denoising Diffusion Implicit Models. ICLR 2021 - from memory, verify.
* Nichol, Dhariwal. Improved Denoising Diffusion Probabilistic Models (cosine schedule). ICML 2021 - from memory, verify.
