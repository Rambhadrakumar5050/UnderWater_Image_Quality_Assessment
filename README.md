# Underwater-Image-Quality-Ranking — TwiceMix

📘 **TwiceMix — Image Quality Ranking**
A reproducible Jupyter Notebook project that trains a VGG-based regressor to predict image quality using paired **High-Quality (HQ)** / **Low-Quality (LQ)** images and synthetic mixes (interpolations). This repository demonstrates on-the-fly mixing, dataset mapping of triplets, correlation-based evaluation (SRCC / KRCC), and an inference helper for single-image scoring.

---

## 🔍 Project overview

This notebook builds a dataset of triplets `(original, high-quality, low-quality)` (CSV mapping), creates synthetic interpolations between HQ and LQ images using mixing coefficient `K`, and trains a `VGGRanker` model (VGG16 backbone + small FC head) to regress a scalar quality score. Evaluation measures rank-consistency across mixes using Spearman and Kendall correlation metrics. The notebook includes utilities to create mappings, train, checkpoint, evaluate, visualize, and infer.

**Key goals**

* Provide a reproducible notebook pipeline to train a quality regressor using synthetic interpolation supervision.
* Produce per-image CSV outputs and aggregated SRCC / KRCC for evaluation.
* Expose a `predict_quality(image_path)` helper to score new images.

---

## ⚙️ Installation

1. Create & activate a virtual environment (recommended)

macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**

```
torch
torchvision
numpy
pandas
pillow
scipy
matplotlib
tqdm
```

> If you set `pretrained=True` for the VGG backbone, your runtime must be able to download torchvision pretrained weights (internet) or you must provide local weights.

---

## 🧹 Data & Preprocessing

**Expected dataset layout**
(The notebook examples use `/kaggle/input/image-underwater-11`; change `DATA_ROOT` as needed.)

```
<DATA_ROOT>/
  train/
    original/
    high-quality/
    low-quality/
  eval/
    original/
    high-quality/
    low-quality/
```

**Mapping CSV format** (no header; one triplet per row)

```
orig_filename.jpg,high_quality_name.jpg,low_quality_name.jpg
```

Filenames are *relative* to the corresponding split subfolder (e.g., `train/original/`).

**Notebook utilities for mapping**

* `create_triplet_mapping_by_id(...)` — create mapping CSVs from folder structure and ID matching rules used in the notebook.

**Example mapping filenames used in the notebook**

* `mapping_train_triplets.csv`
* `mapping_eval_triplets.csv`
* `mapping_test_triplets.csv`

---

## 🧠 Models & key components (as implemented in the notebook)

* **`MappedTwiceMixDataset`** — dataset class that reads mapping CSV rows and returns tensors for `(original, high-quality, low-quality)`.
* **`VGGRanker`** — VGG16 backbone (from `torchvision`) followed by global pooling and an FC head producing a single scalar quality score.

**Mixing utilities**

* `sample_Ks(...)` — sample mixing coefficients for training pairs.
* `mix_images(hq, lq, k)` — produce blended image:
  `mixed = (1 - k) * hq + k * lq` (computed on-the-fly).

**Loss & training**

* `paper_margin_loss(...)` — margin/pairwise style loss implemented in the notebook.
* `train_one_epoch(...)`, `train_full(...)` — training loop functions with checkpointing.

**Evaluation & visualization**

* `evaluate_synthetic(...)` — evaluate model on a set of Ks, writes per-image CSV (`synthetic_test_scores.csv`) and returns aggregated SRCC / KRCC metrics.
* `compute_srcc_krcc_list(...)`, `aggregate_scores_matrix(...)` — helpers for metric computation.
* `visualize_sample_from_mapped(...)` — unnormalize & plot original/HQ/LQ and mixed images.

**Inference**

* `predict_quality(image_path)` — loads an image, applies transforms, and returns a predicted scalar quality (used in a short example inside the notebook).

---

## 🏋️ Training (notebook flow & defaults)

The notebook supplies utilities and example calls. **Defaults found in the notebook:**

* `DATA_ROOT = "/kaggle/input/image-underwater-11"`
* `DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"`
* `LR = 1e-6`
* `BATCH_SIZE = 1`
* `EPOCHS = 20` (cells use `epochs=1` for smoke runs)
* `MIN_K_DIFF = 0.1` (K sampling constraint)
* `MARGIN_EPS = 0.5` (paper epsilon)
* `KS_TEST = [0.0, 0.2, 0.4, 0.6, 0.8]`
* `SEED = 0` (the notebook calls `set_seed(SEED)` for reproducibility)

**Typical notebook calls (copy/paste)**

**Smoke training (1 epoch — run to verify pipeline)**

```python
model, history = train_full(
    root_split_dir="/kaggle/input/image-underwater-11/train",
    mapping_csv="/kaggle/working/mapping_train_triplets.csv",
    save_dir="runs/twice_mix_smoke",
    device=DEVICE,
    epochs=1,
    lr=1e-6,
    pretrained=False,
    sampler="uniform"
)
```

**Full training (example)**

```python
model, history = train_full(
    root_split_dir="/kaggle/input/image-underwater-11/train",
    mapping_csv="/kaggle/working/mapping_train_triplets.csv",
    save_dir="runs/twice_mix_runs",
    device=DEVICE,
    epochs=20,
    lr=1e-6,
    pretrained=False,
    sampler="uniform"
)
```

---

## 📈 Evaluation

Use `evaluate_synthetic(...)` to compute per-image predictions for different mixing coefficients `ks` and aggregate SRCC / KRCC scores.

```python
results = evaluate_synthetic(
    model=model,
    root_split_dir="/kaggle/input/image-underwater-11/eval",
    mapping_csv="/kaggle/working/mapping_eval_triplets.csv",
    device=DEVICE,
    ks=[0.0, 0.2, 0.4, 0.6, 0.8],
    batch_size=1
)
print(results)  # dict with SRCC/KRCC mean & std
```

The notebook writes a CSV `synthetic_test_scores.csv` with per-image scores across Ks for deeper analysis/plots.

---

## 💾 Saving artifacts (required for inference / reproducibility)

* **Checkpoints:** `runs/.../ckpt_epoch{e}.pt` (examples in notebook: `ckpt_epoch1.pt`, `ckpt_epoch40.pt`, etc.)
* **Mapping CSVs generated/used:** `mapping_train_triplets.csv`, `mapping_eval_triplets.csv`
* **Synthetic evaluation CSV:** `synthetic_test_scores.csv`

It is recommended to save the model checkpoint and the mapping CSV used in the run so `predict_quality` and evaluation reproduce the same setup.

---
