from pathlib import Path

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
        print("Нет примеров для показа.")
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

    print(f"Найдено {len(df_pair)} примеров: {true_label} -> {predicted_label}")
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
        print("Все предсказания сохранены в:", save_all_csv_path)

    if save_review_csv_path is not None:
        Path(save_review_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_suspicious.to_csv(save_review_csv_path, index=False)
        print("Подозрительные примеры сохранены в:", save_review_csv_path)

    print("Всего train-примеров проверено:", len(df_all))
    print("Подозрительных примеров найдено:", len(df_suspicious))

    return audit_dataset, df_all, df_suspicious


def show_review_examples(review_df, dataset, start_pos=0, n_rows=3, n_cols=4):
    total = n_rows * n_cols
    part = review_df.iloc[start_pos:start_pos + total]

    if len(part) == 0:
        print("Нет примеров для показа.")
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
