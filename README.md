# Investigating Visual Cues in Saudi Date Fruit Classification Using CNNs

Course project — CAI 421 Deep Learning, Princess Nourah bint Abdulrahman University
Authors: Aisha Almutairi, Ravef Alqarni, Aleen Alshehri

## Overview
We study which visual cue — **color, shape or texture** — convolutional neural networks rely on when
classifying nine Saudi date varieties, and whether attention plus cue-balanced training reduces that reliance.

| Model | Description |
|---|---|
| B1 | Re-implementation of the 4-layer CNN of Alhamdan & Howe (2021), configuration 48-4L (120×120, SGD lr 0.001, momentum 0.9, batch 32, up to 200 epochs, early stopping) |
| B2 | MobileNetV3-Large (ImageNet), head trained 5 epochs (Adam 1e-3), then fine-tuned end-to-end (Adam 1e-5, early stopping) |
| **Proposed** | B2 + **CBAM** attention (reduction 16) between backbone and pooling, trained with **cue-balanced augmentation** (random grayscale, 4×4 patch-shuffle and Gaussian blur, p = 0.2 each), same schedule as B2 |

Cue-isolation test sets are built from the 502 test images: **grayscale** (no color), **silhouette** (shape only),
**4×4 patch-shuffle** (no global shape) and **Gaussian blur σ = 4** (no fine texture).
Grad-CAM maps show where B2 and the proposed model look.

## Results (test set, 502 images, one run, seed 42, Tesla T4)

| Model | Accuracy | Macro-F1 | Params (M) | Train (min) | Inference (ms/img) |
|---|---|---|---|---|---|
| B1 — plain SGD (no momentum) | 0.6753 | 0.6499 | 35.03 | 7.4 | – |
| **B1 — SGD momentum 0.9** | **0.8884** | **0.8866** | 35.03 | 6.7 | 0.54 |
| B2 — fine-tune lr 1e-4 | 0.8884 | 0.8863 | 3.01 | 5.6 | – |
| **B2 — fine-tune lr 1e-5** | **0.9323** | **0.9334** | 3.01 | 2.4 | 0.80 |
| **Proposed — MobileNetV3 + CBAM + cue aug.** | **0.9323** | **0.9353** | 3.12 | 6.2 | 0.82 |

### Cue-isolation accuracy (%)

| Model | Clean | Grayscale | Silhouette | Patch-shuffle | Blur |
|---|---|---|---|---|---|
| B1 | 88.84 | 17.53 | 18.33 | 17.73 | 88.84 |
| B2 | 93.23 | 37.05 | 11.55 | 54.58 | 22.91 |
| Proposed | 92.83 | 86.25 | 18.33 | 74.90 | 75.70 |

Cue-reliance drop ΔAcc = Clean − Variant (points): B1 71.3 / 70.5 / 71.1 / 0.0; B2 56.2 / 81.7 / 38.7 / 70.3;
Proposed 6.6 / 74.5 / 17.9 / 17.1 (grayscale / silhouette / patch-shuffle / blur).

### Comparison with published results on the same dataset
| Study | Accuracy | Source |
|---|---|---|
| Alhamdan & Howe [1], 48-4L | 0.960 | quoted |
| Alhamdan & Howe [1], 70-4L (best) | 0.970 | quoted |
| Adnan & Nasharuddin [2] | 0.942 | quoted |
| Ours: B1 (re-implementation of 48-4L) | 0.888 | reproduced |
| Ours: B2 | 0.932 | ours |
| Ours: Proposed | 0.932 | ours |

## Dataset
*Date Fruit Image Dataset in Controlled Environment* — W. S. N. Alhamdan and J. M. Howe, AMLTA 2021, Springer,
doi:10.1007/978-3-030-69717-4_16. Kaggle, license **CC BY 4.0**. 1,658 images, 9 classes
(Ajwa, Galaxy, Medjool, Meneifi, Nabtat Ali, Rutab, Shaishe, Sokari, Sugaey). We use the fixed train/test split published at
https://github.com/osama-fahd/Saudi-Date-Fruit-Image-Classification (train 1,156 / test 502) and hold out 15 % of
train (stratified, seed 42) for validation (982 / 174 / 502). The exact file lists are written to `splits/` by the notebook.

Expected layout: `dates/train/<Class>/*.jpg`, `dates/test/<Class>/*.jpg`.

## How to reproduce
1. Open `DL_project.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. Put the dataset in Google Drive at `MyDrive/dates/` (or edit `DRIVE_DIR` in the settings cell).
3. Run all cells from top to bottom (about 40 minutes). Metrics (`results/*.json, *.csv`), figures and trained
   models are written to `MyDrive/dates_project/`.

Install locally with `pip install -r requirements.txt`.

## Reproducibility
- Fixed random seed 42 (`keras.utils.set_random_seed`) for the split, initialisation, shuffling and the
  patch-shuffle permutations of the cue tests; the split file lists are exported.
- TensorFlow GPU kernels are not fully deterministic, so a re-run can differ slightly; all numbers above come from
  the single run saved in `DL_project.ipynb`.

## AI tool disclosure
AI tools were used for coding support and language editing, as permitted by the course; all content was reviewed by the authors.
