from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from visualization import imshow
except ImportError:
    from .visualization import imshow


def build_train_audit_dataset(train_files, label_encoder, dataset_cls):
    """Build a train dataset without train-time augmentation for label audit."""
    return dataset_cls(
        files=train_files,
        label_encoder=label_encoder,
        mode="val",
    )


@torch.no_grad()
def find_wrong_label_candidates(
    model,
    dataset,
    label_encoder,
    device,
    batch_size=64,
    num_workers=2,
):
    """Run the model over a dataset and return per-image prediction details."""
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    idx_to_class = {i: cls_name for i, cls_name in enumerate(label_encoder.classes_)}

    records = []
    start_idx = 0

    for x, y in tqdm(loader, desc="Scanning dataset for suspicious labels"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)

        top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
        confs = top2_probs[:, 0]
        preds = top2_indices[:, 0]
        second_confs = top2_probs[:, 1]
        second_preds = top2_indices[:, 1]
        margins = confs - second_confs

        losses = criterion(logits, y)
        batch_size_now = x.size(0)

        for i in range(batch_size_now):
            dataset_index = start_idx + i
            file_path = Path(dataset.files[dataset_index])

            true_idx = int(y[i].item())
            pred_idx = int(preds[i].item())
            second_idx = int(second_preds[i].item())

            records.append(
                {
                    "dataset_index": dataset_index,
                    "path": str(file_path),
                    "filename": file_path.name,
                    "current_label": idx_to_class[true_idx],
                    "predicted_label": idx_to_class[pred_idx],
                    "second_predicted_label": idx_to_class[second_idx],
                    "confidence": float(confs[i].item()),
                    "second_confidence": float(second_confs[i].item()),
                    "margin": float(margins[i].item()),
                    "loss": float(losses[i].item()),
                    "is_mismatch": pred_idx != true_idx,
                }
            )

        start_idx += batch_size_now

    return pd.DataFrame(records)


def select_suspicious_examples(
    df_all,
    min_confidence=0.90,
    min_margin=0.20,
    top_n=300,
):
    """Select confident model-label disagreements for manual review."""
    df_suspicious = df_all[df_all["is_mismatch"] == True].copy()
    df_suspicious = df_suspicious[
        (df_suspicious["confidence"] >= min_confidence)
        & (df_suspicious["margin"] >= min_margin)
    ].copy()

    return df_suspicious.sort_values(
        by=["confidence", "margin", "loss"],
        ascending=[False, False, False],
    ).head(top_n).reset_index(drop=True)


def show_suspicious_examples(df, dataset, n_rows=3, n_cols=4, start_pos=0):
    total = n_rows * n_cols
    part = df.iloc[start_pos:start_pos + total].reset_index(drop=True)

    if len(part) == 0:
        print("No examples to show.")
        return

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i in range(len(part)):
        row = part.iloc[i]
        dataset_index = int(row["dataset_index"])
        image_tensor, _ = dataset[dataset_index]

        title = (
            f'true: {row["current_label"]}\n'
            f'pred: {row["predicted_label"]}\n'
            f'conf: {row["confidence"]:.3f} | loss: {row["loss"]:.3f}'
        )

        imshow(image_tensor, title=title, plt_ax=axes[i])

    plt.tight_layout()
    plt.show()


def show_class_pair_confusions(
    df_all,
    true_label,
    predicted_label,
    dataset,
    n_rows=3,
    n_cols=4,
    start_pos=0,
):
    df_pair = df_all[
        (df_all["current_label"] == true_label)
        & (df_all["predicted_label"] == predicted_label)
    ].copy()

    df_pair = df_pair.sort_values(
        by=["confidence", "margin", "loss"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print(f"Found {len(df_pair)} examples: {true_label} -> {predicted_label}")
    show_suspicious_examples(
        df_pair,
        dataset,
        n_rows=n_rows,
        n_cols=n_cols,
        start_pos=start_pos,
    )
    return df_pair


def run_label_audit(
    model,
    train_files,
    label_encoder,
    dataset_cls,
    device,
    batch_size=64,
    num_workers=0,
    min_confidence=0.90,
    min_margin=0.20,
    top_n=300,
    save_all_csv_path=None,
    save_review_csv_path=None,
):
    audit_dataset = build_train_audit_dataset(
        train_files=train_files,
        label_encoder=label_encoder,
        dataset_cls=dataset_cls,
    )

    df_all = find_wrong_label_candidates(
        model=model,
        dataset=audit_dataset,
        label_encoder=label_encoder,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    df_suspicious = select_suspicious_examples(
        df_all=df_all,
        min_confidence=min_confidence,
        min_margin=min_margin,
        top_n=top_n,
    )

    if save_all_csv_path is not None:
        Path(save_all_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(save_all_csv_path, index=False)
        print("All predictions saved to:", save_all_csv_path)

    if save_review_csv_path is not None:
        Path(save_review_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_suspicious.to_csv(save_review_csv_path, index=False)
        print("Suspicious examples saved to:", save_review_csv_path)

    print("Total train examples scanned:", len(df_all))
    print("Suspicious examples found:", len(df_suspicious))

    return audit_dataset, df_all, df_suspicious


def show_review_examples(review_df, dataset, start_pos=0, n_rows=3, n_cols=4):
    total = n_rows * n_cols
    part = review_df.iloc[start_pos:start_pos + total]

    if len(part) == 0:
        print("No examples to show.")
        return

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, (row_index, row) in zip(axes, part.iterrows()):
        dataset_index = int(row["dataset_index"])
        image_tensor, _ = dataset[dataset_index]

        title = (
            f"row: {row_index}\n"
            f"folder: {row['current_label']}\n"
            f"model: {row['predicted_label']}\n"
            f"conf: {row['confidence']:.3f} | margin: {row['margin']:.3f}"
        )

        imshow(image_tensor, title=title, plt_ax=ax)

    plt.tight_layout()
    plt.show()


def _resolve_review_path(path_value, dataset_root=None):
    src_path = Path(path_value)

    if src_path.exists() or dataset_root is None:
        return src_path

    dataset_root = Path(dataset_root)
    path_parts = src_path.parts

    for marker in ("train", "simpsons_dataset"):
        if marker in path_parts:
            marker_index = path_parts.index(marker)
            candidate = dataset_root / Path(*path_parts[marker_index + 1:])
            if candidate.exists():
                return candidate

    candidate = dataset_root / src_path.parent.name / src_path.name
    if candidate.exists():
        return candidate

    return src_path


def move_reviewed_files_to_class(review_df, dry_run=True, dataset_root=None):
    """
    Moves only rows with action == "MOVE".
    The target class comes from correct_label when available, otherwise from predicted_label.
    """
    required_columns = ["path", "predicted_label", "action"]
    missing_columns = [col for col in required_columns if col not in review_df.columns]
    if missing_columns:
        raise ValueError(f"review_df is missing required columns: {missing_columns}")

    actions = review_df["action"].fillna("").astype(str).str.upper()
    rows_to_move = review_df[actions == "MOVE"].copy()
    print("Files to move:", len(rows_to_move))

    moved_paths = {}

    for _, row in rows_to_move.iterrows():
        src_path = _resolve_review_path(row["path"], dataset_root=dataset_root)
        correct_label = row.get("correct_label", "")
        target_class = "" if pd.isna(correct_label) else str(correct_label).strip()
        if not target_class:
            target_class = str(row["predicted_label"]).strip()

        if not src_path.exists():
            print("File not found, skipping:", src_path)
            continue

        current_class_dir = src_path.parent
        split_root_dir = current_class_dir.parent
        dst_dir = split_root_dir / target_class
        dst_path = dst_dir / src_path.name

        if not dst_dir.exists():
            print("Target class directory not found, skipping:", dst_dir)
            continue

        if current_class_dir.name == target_class:
            print("File is already in the target directory, skipping:", src_path)
            continue

        if dst_path.exists():
            stem = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while dst_path.exists():
                dst_path = dst_dir / f"{stem}_moved_{counter}{suffix}"
                counter += 1

        print("FROM:", src_path)
        print("TO:  ", dst_path)

        if not dry_run:
            shutil.move(str(src_path), str(dst_path))
            moved_paths[src_path] = dst_path

    if dry_run:
        print("This was dry_run=True. No files were moved.")
    else:
        print("Done. Files were moved.")

    return moved_paths
