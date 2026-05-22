# Experiments

This log summarizes the modeling progression used in the notebook. The goal was not only to maximize the leaderboard score, but also to keep the workflow interpretable for review.

| Run | Model | Main Change | Validation Macro F1 | Validation Accuracy | Kaggle Score | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 | SimpleCNN | Train from scratch | `0.737` | `0.839` | baseline only | Confirms that the data pipeline learns useful visual patterns, but model capacity is limited. |
| 2 | EfficientNetV2-S | Classifier-head training + fine-tuning | `0.829` | `0.935` | `0.972` | Used as the main walkthrough model because the learning curves show gradual optimization. |
| 3 | DenseNet121 | No-freeze fine-tuning with discriminative learning rates | `0.935` | `0.981` | `0.993` | Best final model by validation macro F1, validation accuracy, and Kaggle score. |

## Interpretation

The SimpleCNN baseline provides a sanity check: the train/validation split, labels, transforms, and dataloaders are usable. Transfer learning then gives the largest performance gain because ImageNet-pretrained backbones already contain strong visual features.

EfficientNetV2-S remains the clearest model for explaining the training process because its metrics improve gradually across stages. DenseNet121 is reported as the best final model because it reaches the strongest validation and Kaggle metrics, even though its learning curve saturates quickly.

## Error Analysis

The lowest-F1 classes are mostly rare characters with very small validation support. This makes individual per-class metrics noisy: a single mistake can drive F1 to zero when a class has only one or two validation examples. For this reason, macro F1 is useful for exposing rare-class weaknesses, but rare-class scores should always be interpreted together with `support`.
