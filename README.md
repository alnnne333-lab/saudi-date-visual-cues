# Investigating Visual Cues in Saudi Date Fruit Classification Using CNNs

Course project — CAI 421 Deep Learning, Princess Nourah bint Abdulrahman University
Authors: Aisha Almutairi, Ravef Alqarni, Aleen Alshehri

## Overview
We study which visual cue — **color, shape or texture** — convolutional neural networks rely on
when classifying nine Saudi date varieties. This repository currently contains the
**Checkpoint 2 baselines**:

| Model | Description |
|---|---|
| B1 | Re-implementation of the 4-layer CNN of Alhamdan & Howe (2021), configuration 48-4L (120×120 input, SGD lr 0.001 with momentum 0.9, batch 32, 200 epochs) |
| B2 | MobileNetV3-Large (ImageNet), head trained 5 epochs (Adam 1e-3), then fine-tuned end-to-end (Adam 1e-5, early stopping); no attention, no cue-balanced augmentation |

## Results (test set, 502 images)

| Model | Accuracy | Macro-F1 | Params (M) | Train time (min) | Inference (ms/img) |
|---|---|---|---|---|---|
| B1 — plain SGD (no momentum) | 0.671 | 0.636 | 35.03 | 7.8 | – |
| **B1 — SGD momentum 0.9** | **0.888** | **0.888** | 35.03 | 7.5 | 0.60 |
| B2 — fine-tune lr 1e-4 | 0.888 | 0.886 | 3.01 | 4.1 | – |
| **B2 — fine-tune lr 1e-5** | **0.932** | **0.933** | 3.01 | 2.3 | 0.87 |
| Alhamdan & Howe [1], 48-4L (quoted) | 0.960 | – | – | – | – |
| Alhamdan & Howe [1], 70-4L (quoted) | 0.970 | – | – | – | – |
| Adnan & Nasharuddin [2] (quoted) | 0.942 | – | – | – | – |

Hardware: Google Colab, NVIDIA Tesla T4. Bold rows are the baselines used in the paper.

Planned next: MobileNetV3 + CBAM with cue-balanced augmentation, cue-isolation test sets
(grayscale, silhouette, patch-shuffle, blur) and Grad-CAM analysis.

## Dataset
*Date Fruit Image Dataset in Controlled Environment* — W. S. N. Alhamdan and J. M. Howe,
AMLTA 2021, Springer, doi:10.1007/978-3-030-69717-4_16. Kaggle, license **CC BY 4.0**.
1,658 images, 9 classes (Ajwa, Galaxy, Medjool, Meneifi, Nabtat Ali, Rutab, Shaishe, Sokari, Sugaey).
We use the fixed train/test split published at
https://github.com/osama-fahd/Saudi-Date-Fruit-Image-Classification
(train 1,156 / test 502) and hold out 15 % of train (stratified, seed 42) for validation.
The exact file lists are saved in `splits/` when the notebook runs.

Expected folder layout:
```
dates/
  train/<Class>/*.jpg
  test/<Class>/*.jpg
```

## How to reproduce
1. Open `date_fruit_baselines.ipynb` in Google Colab and select a **GPU** runtime.
2. Put the dataset in Google Drive at `MyDrive/dates/` (or edit `DRIVE_DIR` in the setup cell).
3. Run all cells. Results (`results/*.json, *.csv`), figures and trained models are written to
   `MyDrive/dates_project/`.

Install locally with `pip install -r requirements.txt`.

## Reproducibility
- Fixed random seed 42 (`keras.utils.set_random_seed`) for the split, weight initialisation and shuffling.
- The validation/test file lists are exported to `splits/`.
- GPU kernels in TensorFlow are not fully deterministic, so repeated runs can differ slightly
  (typically well under one percentage point).

## AI tool disclosure
AI tools were used for coding support and language editing, as permitted by the course;
all content was reviewed by the authors.
