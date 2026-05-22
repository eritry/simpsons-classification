# Simpsons Character Classification

Portfolio-ready computer vision project for classifying characters from *The Simpsons*.

The project demonstrates an end-to-end PyTorch image-classification workflow: dataset loading, stratified validation, class-imbalance handling, transfer learning, fine-tuning, error analysis, and Kaggle-style submission generation.

## Highlights

- Built an end-to-end PyTorch classification pipeline for 42 imbalanced character classes.
- Improved validation macro F1 from a SimpleCNN baseline of `0.737` to DenseNet121 macro F1 of `0.935`.
- Used transfer learning, weighted sampling, class-weighted loss, and manual label-audit workflow.
- Reached a Kaggle score of `0.993` with the best final DenseNet121 run.
- Organized the notebook into reusable modules for reproducible Colab execution.

## What This Shows

- Transfer learning with ImageNet-pretrained DenseNet121 and EfficientNetV2-S.
- Baseline comparison across SimpleCNN, DenseNet121, and EfficientNetV2-S.
- Clean train/validation split with stratification.
- Class-imbalance handling with weighted sampling and weighted loss.
- Modular notebook design with reusable Python helpers.
- Error-analysis workflow for identifying suspicious labels.
- Reproducible Colab setup using GitHub-hosted source code and dataset release assets.

## Model Comparison

| Model | Pretraining | Parameters | Validation Macro F1 | Validation Accuracy | Kaggle Score | Role |
|---|---:|---:|---:|---:|---:|---|
| SimpleCNN | No | `180,762` | `0.737` | `0.839` | baseline only | From-scratch baseline |
| DenseNet121 | ImageNet | `7.2M` | `0.935` | `0.981` | `0.993` | Best final model |
| EfficientNetV2-S | ImageNet | `21M` | `0.829` | `0.935` | `0.972` | Main training-curve walkthrough |

The baseline confirms that the data pipeline learns meaningful visual features, while the transfer-learning models show the performance gain from pretrained representations. EfficientNetV2-S is the default walkthrough model because its learning curves are easier to inspect, while DenseNet121 is reported as the strongest final model. Macro F1 is the primary validation metric because the class distribution is highly imbalanced. The same comparison is versioned in `artifacts/model_comparison.csv`.

## Repository Structure

- `simpsons.ipynb` - main Colab notebook with data loading, training, validation, audit, and submission steps.
- `data_io.py` - dataset download and extraction helpers for Colab.
- `dataset.py` - dataset class, transforms, stratified split helpers, and dataloader construction.
- `model.py` - SimpleCNN, DenseNet121, and EfficientNetV2-S model factories.
- `training.py` - training loop, metrics, checkpointing, history plotting, and prediction helpers.
- `visualization.py` - image display, prediction visualization, and class-distribution plots.
- `label_audit.py` - helpers for finding suspicious labels, reviewing audit examples, and applying reviewed label moves.
- `submission.py` - checkpoint loading and Kaggle-style submission helpers.
- `artifacts/label_audit/suspicious_manual.csv` - manually reviewed suspicious-label candidates used by the optional label-fix step.
- `artifacts/model_comparison.csv` - compact model-comparison table used by the README and notebook discussion.
- `requirements.txt` - minimal Python dependencies for local or Colab-style execution.

## Data and Artifacts

All Google Drive artifacts are configured to live under:

```text
/content/drive/MyDrive/Colab Notebooks/simpsons
```

Code modules are loaded from the GitHub repository cloned into:

```text
/content/simpsons-classification
```

The notebook expects the Kaggle archive to be unpacked in Colab into `/content/train` and `/content/testset`.
The dataset archive is published as a GitHub Release asset:

```text
https://github.com/eritry/simpsons-classification/releases/tag/dataset-v1
```

The notebook caches that archive on Google Drive at:

```text
/content/drive/MyDrive/Colab Notebooks/simpsons/dataset/journey-springfield.zip
```

The manually reviewed label-audit CSV is versioned in the repository:

```text
artifacts/label_audit/suspicious_manual.csv
```

Rows marked with `MOVE` are consumed by the optional label-fix step in the notebook. The CSV stores Colab-style image paths; the helper resolves them against the current extracted training directory before moving files.

## How to Run

Open in Colab:

```text
https://colab.research.google.com/github/eritry/simpsons-classification/blob/main/simpsons.ipynb
```

Run the notebook top to bottom. The first run downloads and caches the dataset archive; later runs reuse the cached copy from Google Drive.

For a local environment, install the minimal dependencies with:

```bash
pip install -r requirements.txt
```

## Experiment Setup

- Image size: `224x224`
- Default batch size: `64`
- Train/validation split: stratified, `75/25`
- Main validation metric: macro F1
- Label fixes: disabled by default; can be enabled with `APPLY_LABEL_FIXES = True`
- Default walkthrough model: EfficientNetV2-S
- Best final model: DenseNet121

## Final Results

Best final model: DenseNet121

- Validation accuracy: `0.981`
- Validation macro F1: `0.935`
- Kaggle score recorded during the project: `0.993`

Main walkthrough model: EfficientNetV2-S

- Validation accuracy: `0.935`
- Validation macro F1: `0.829`
- Kaggle score recorded during the project: `0.972`

Exact numbers can vary slightly with runtime, random seed, and future label-audit decisions.

## Notes

The notebook is intentionally structured as a readable project report rather than a single monolithic script. Reusable code lives in `.py` modules, while the notebook focuses on experiment decisions, validation, and interpretation.
