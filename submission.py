import pandas as pd
import torch


def load_best_model_if_exists(model, checkpoint_path, device):
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
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
