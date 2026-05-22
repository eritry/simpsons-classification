import pandas as pd
import torch
from pathlib import Path


def load_model_weights(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_best_model_if_exists(model, checkpoint_path, device):
    if checkpoint_path.exists():
        model = load_model_weights(model, checkpoint_path, device)
    else:
        print(f"Best model checkpoint not found: {checkpoint_path}. Using current model state.")

    return model


def create_submission(test_files, predicted_labels, sample_submission_path=None):
    sample_submission = None

    if sample_submission_path is not None:
        sample_submission = pd.read_csv(sample_submission_path)

    submission = pd.DataFrame({
        "Id": [path.name for path in test_files],
        "Expected": predicted_labels,
    })

    return submission, sample_submission
