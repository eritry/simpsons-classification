import numpy as np
import torch
import torch.nn as nn
from matplotlib import patches, pyplot as plt
from matplotlib.font_manager import FontProperties


def format_class_name(class_name):
    return " ".join(part.capitalize() for part in class_name.split("_"))


def imshow(
    inp,
    title=None,
    plt_ax=plt,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Show a normalized CHW image tensor."""
    if isinstance(inp, torch.Tensor):
        inp = inp.detach().cpu().numpy()

    inp = inp.transpose((1, 2, 0))
    inp = np.array(std) * inp + np.array(mean)
    inp = np.clip(inp, 0, 1)

    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_images_from_loader(n_rows, n_cols, loader, label_encoder):
    n_images = n_rows * n_cols
    images_list = []
    labels_list = []
    loader_iter = iter(loader)

    while len(images_list) < n_images:
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, labels = next(loader_iter)

        for image, label in zip(images, labels):
            images_list.append(image)
            labels_list.append(label)

            if len(images_list) >= n_images:
                break

    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        sharey=True,
        sharex=True,
    )

    ax = np.array([ax]) if n_rows == 1 and n_cols == 1 else ax.flatten()

    for fig_x, im_val, label in zip(ax, images_list, labels_list):
        label = int(label.item()) if hasattr(label, "item") else int(label)
        class_name = label_encoder.inverse_transform([label])[0]

        imshow(
            im_val,
            title=format_class_name(class_name),
            plt_ax=fig_x,
        )

        fig_x.set_axis_off()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_images_with_predictions(
    n_rows,
    n_cols,
    dataset,
    model,
    label_encoder,
    device,
):
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        sharey=True,
        sharex=True,
    )

    axes = np.array(axs).reshape(-1)

    for fig_x in axes:
        random_index = int(np.random.uniform(0, len(dataset)))
        image_tensor, label = dataset[random_index]

        true_class_name = label_encoder.inverse_transform([label])[0]
        true_label = format_class_name(true_class_name)
        imshow(image_tensor, title=true_label, plt_ax=fig_x)

        logits = model(image_tensor.unsqueeze(0).to(device))
        prob_pred = nn.functional.softmax(logits, dim=-1).cpu().numpy()

        predicted_proba = np.max(prob_pred) * 100
        predicted_idx = int(np.argmax(prob_pred))
        predicted_class_name = label_encoder.inverse_transform([predicted_idx])[0]
        predicted_label = format_class_name(predicted_class_name)
        predicted_text = f"{predicted_label}:\n {predicted_proba:.1f}%"

        font = FontProperties().copy()
        fig_x.add_patch(
            patches.Rectangle(
                (0, 190),
                7 * len(predicted_label),
                25,
                color="white",
            )
        )
        fig_x.text(
            2,
            195,
            predicted_text,
            horizontalalignment="left",
            fontproperties=font,
            verticalalignment="top",
            fontsize=8,
            color="black",
            fontweight="bold",
        )
        fig_x.set_axis_off()

