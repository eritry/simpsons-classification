# Simpsons Image Classification

Image classification notebook for the Simpsons character dataset.

The project contains:

- `simpsons.ipynb` - main Colab notebook with data loading, training, validation, audit, and submission steps.
- `utils.py` - training loop, metrics, checkpointing, plotting history, and prediction helpers.
- `visualization.py` - image display helpers.
- `label_audit.py` - helpers for finding suspicious labels and reviewing audit examples.

All Google Drive artifacts are configured to live under:

```text
/content/drive/MyDrive/Colab Notebooks/simpsons
```

Code modules are loaded from the GitHub repository cloned into:

```text
/content/simpsons-classification
```

The notebook expects the Kaggle archive to be unpacked in Colab into `/content/train` and `/content/testset`.

Open in Colab:

```text
https://colab.research.google.com/github/eritry/simpsons-classification/blob/main/simpsons.ipynb
```
