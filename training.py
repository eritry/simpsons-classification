import json
import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import get_backbone, get_classifier


def make_training_history():
    return {
        "train_acc_step": [],
        "valid_acc_step": [],
        "train_f1_step": [],
        "valid_f1_step": [],
        "train_acc_epoch": [],
        "valid_acc_epoch": [],
        "train_f1_epoch": [],
        "valid_f1_epoch": [],
        "train_loss_step": [],
        "train_loss_epoch": [],
    }


def _ensure_history_keys(history):
    default_history = make_training_history()

    for key, value in default_history.items():
        if key not in history:
            history[key] = value

    return history


def _get_num_classes(model, dataloaders=None):
    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        return int(model.fc.out_features)

    if hasattr(model, "classifier"):
        classifier = model.classifier

        if hasattr(classifier, "out_features"):
            return int(classifier.out_features)

        if isinstance(classifier, torch.nn.Sequential):
            for layer in reversed(classifier):
                if hasattr(layer, "out_features"):
                    return int(layer.out_features)

    if dataloaders is not None and "train" in dataloaders:
        labels = []
        dataset = dataloaders["train"].dataset

        if hasattr(dataset, "label_encoder"):
            return len(dataset.label_encoder.classes_)

        if hasattr(dataset, "classes"):
            return len(dataset.classes)

        for _, y in dataloaders["train"]:
            labels.extend(y.detach().cpu().numpy().tolist())
            break

        if len(labels) > 0:
            return int(max(labels) + 1)

    raise ValueError(
        "Could not infer the number of classes. "
        "Pass a model with model.fc.out_features or a dataset with label_encoder/classes."
    )


def _macro_f1(y_true, y_pred, num_classes):
    labels = list(range(num_classes))

    return float(
        f1_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def evaluate(model, dataloader, device, num_classes=None):
    model.eval()

    correct = 0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())

    accuracy = correct / total if total > 0 else 0.0

    if num_classes is None:
        if len(all_true) == 0 and len(all_pred) == 0:
            num_classes = 0
        else:
            num_classes = int(max(all_true + all_pred) + 1)

    macro_f1 = _macro_f1(
        y_true=all_true,
        y_pred=all_pred,
        num_classes=num_classes,
    ) if num_classes > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def save_training_state(
    save_path,
    model,
    optimizer,
    epoch,
    step,
    history,
    best_valid_f1,
):
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_valid_f1": best_valid_f1,
    }

    torch.save(checkpoint, save_path)


def save_history_json(history, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_model_and_history(
    save_dir,
    checkpoint_name,
    history_name,
    model,
    optimizer,
    epoch,
    step,
    history,
    best_valid_f1,
):
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    history_path = os.path.join(save_dir, history_name)

    save_training_state(
        save_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        history=history,
        best_valid_f1=best_valid_f1,
    )

    save_history_json(history, history_path)


def _format_metric(value):
    return f"{value:.4f}"


def _best_history_value(history, key):
    values = history.get(f"{key}_step", []) + history.get(f"{key}_epoch", [])
    return max(values) if values else 0.0


def _format_epoch_row(
    epoch,
    num_epochs,
    train_loss,
    train_acc,
    valid_acc,
    train_f1,
    valid_f1,
    best_valid_f1,
    status,
):
    return (
        f"{epoch}/{num_epochs:<5}"
        f"{_format_metric(train_loss):<12}"
        f"{_format_metric(train_acc):<11}"
        f"{_format_metric(valid_acc):<9}"
        f"{_format_metric(train_f1):<10}"
        f"{_format_metric(valid_f1):<8}"
        f"{_format_metric(best_valid_f1):<9}"
        f"{status}"
    )


def epoch_history_rows(history):
    rows = []
    train_losses = history.get("train_loss_epoch", [])
    train_accs = history.get("train_acc_epoch", [])
    valid_accs = history.get("valid_acc_epoch", [])
    train_f1s = history.get("train_f1_epoch", [])
    valid_f1s = history.get("valid_f1_epoch", [])

    best_valid_f1 = 0.0

    for index, train_loss in enumerate(train_losses):
        valid_f1 = valid_f1s[index]
        checkpoint = "saved" if valid_f1 > best_valid_f1 else "kept"
        best_valid_f1 = max(best_valid_f1, valid_f1)

        rows.append(
            {
                "epoch": index + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accs[index]),
                "valid_accuracy": float(valid_accs[index]),
                "train_macro_f1": float(train_f1s[index]),
                "valid_macro_f1": float(valid_f1),
                "best_valid_macro_f1": float(best_valid_f1),
                "checkpoint": checkpoint,
            }
        )

    return rows


def train_CNN(
    model,
    num_epochs,
    dataloaders,
    optimizer,
    loss_func,
    device,
    val_every_steps=None,
    save_dir="/content/drive/MyDrive/checkpoints",
    scheduler=None,
    history=None,
    stage_name="training",
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    if history is None:
        history = make_training_history()
    else:
        history = _ensure_history_keys(history)

    num_classes = _get_num_classes(model, dataloaders=dataloaders)

    if len(history["valid_f1_step"]) > 0:
        best_valid_f1 = max(history["valid_f1_step"])
    elif len(history["valid_f1_epoch"]) > 0:
        best_valid_f1 = max(history["valid_f1_epoch"])
    else:
        best_valid_f1 = 0.0

    steps = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    tqdm.write(f"\n=== {stage_name} ===")
    tqdm.write(f"Epochs: {num_epochs} | Checkpoint metric: valid_f1")
    tqdm.write("")
    tqdm.write(
        f"{'Epoch':<7}"
        f"{'Train Loss':<12}"
        f"{'Train Acc':<11}"
        f"{'Val Acc':<9}"
        f"{'Train F1':<10}"
        f"{'Val F1':<8}"
        f"{'Best F1':<9}"
        f"{'Status'}"
    )

    for epoch in range(num_epochs):
        model.train()

        train_correct_step = 0
        train_total_step = 0
        train_true_step = []
        train_pred_step = []

        train_correct_epoch = 0
        train_total_epoch = 0
        train_loss_sum_epoch = 0.0
        train_true_epoch = []
        train_pred_epoch = []

        progress = tqdm(
            dataloaders["train"],
            desc=f"epoch {epoch + 1:02d}/{num_epochs:02d}",
            dynamic_ncols=True,
            leave=False,
        )

        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            batch_size = y.size(0)
            progress.set_postfix(loss=f"{loss_value:.4f}")

            history["train_loss_step"].append(loss_value)
            train_loss_sum_epoch += loss_value * batch_size

            preds = torch.argmax(outputs, dim=1)
            correct_batch = (preds == y).sum().item()

            train_correct_step += correct_batch
            train_total_step += batch_size

            train_correct_epoch += correct_batch
            train_total_epoch += batch_size

            y_cpu = y.detach().cpu().numpy().tolist()
            preds_cpu = preds.detach().cpu().numpy().tolist()

            train_true_step.extend(y_cpu)
            train_pred_step.extend(preds_cpu)

            train_true_epoch.extend(y_cpu)
            train_pred_epoch.extend(preds_cpu)

            steps += 1

            if val_every_steps is not None and val_every_steps > 0 and steps % val_every_steps == 0:
                train_acc_step = train_correct_step / train_total_step if train_total_step > 0 else 0.0
                train_f1_step = _macro_f1(
                    y_true=train_true_step,
                    y_pred=train_pred_step,
                    num_classes=num_classes,
                )

                valid_metrics = evaluate(
                    model=model,
                    dataloader=dataloaders["val"],
                    device=device,
                    num_classes=num_classes,
                )

                valid_acc_step = valid_metrics["accuracy"]
                valid_f1_step = valid_metrics["macro_f1"]

                history["train_acc_step"].append(float(train_acc_step))
                history["valid_acc_step"].append(float(valid_acc_step))
                history["train_f1_step"].append(float(train_f1_step))
                history["valid_f1_step"].append(float(valid_f1_step))

                tqdm.write(
                    f"  step {steps:04d} | "
                    f"loss {_format_metric(loss_value)} | "
                    f"train_acc {_format_metric(train_acc_step)} | "
                    f"valid_acc {_format_metric(valid_acc_step)} | "
                    f"train_f1 {_format_metric(train_f1_step)} | "
                    f"valid_f1 {_format_metric(valid_f1_step)}"
                )

                if valid_f1_step > best_valid_f1:
                    best_valid_f1 = valid_f1_step

                    save_model_and_history(
                        save_dir=save_dir,
                        checkpoint_name="best_checkpoint.pt",
                        history_name="best_history.json",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=steps,
                        history=history,
                        best_valid_f1=best_valid_f1,
                    )

                    torch.save(
                        model.state_dict(),
                        best_model_path,
                    )

                model.train()

                train_correct_step = 0
                train_total_step = 0
                train_true_step = []
                train_pred_step = []

        train_acc_epoch = train_correct_epoch / train_total_epoch if train_total_epoch > 0 else 0.0
        train_loss_epoch = train_loss_sum_epoch / train_total_epoch if train_total_epoch > 0 else 0.0
        train_f1_epoch = _macro_f1(
            y_true=train_true_epoch,
            y_pred=train_pred_epoch,
            num_classes=num_classes,
        )

        valid_metrics_epoch = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            num_classes=num_classes,
        )

        valid_acc_epoch = valid_metrics_epoch["accuracy"]
        valid_f1_epoch = valid_metrics_epoch["macro_f1"]

        history["train_acc_epoch"].append(float(train_acc_epoch))
        history["valid_acc_epoch"].append(float(valid_acc_epoch))
        history["train_f1_epoch"].append(float(train_f1_epoch))
        history["valid_f1_epoch"].append(float(valid_f1_epoch))
        history["train_loss_epoch"].append(float(train_loss_epoch))

        if scheduler is not None:
            scheduler.step()

        checkpoint_status = "kept"
        if valid_f1_epoch > best_valid_f1:
            best_valid_f1 = valid_f1_epoch
            checkpoint_status = "saved"

            save_model_and_history(
                save_dir=save_dir,
                checkpoint_name="best_checkpoint.pt",
                history_name="best_history.json",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=steps,
                history=history,
                best_valid_f1=best_valid_f1,
            )

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        save_model_and_history(
            save_dir=save_dir,
            checkpoint_name="last_checkpoint.pt",
            history_name="last_history.json",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=steps,
            history=history,
            best_valid_f1=best_valid_f1,
        )

        tqdm.write(
            _format_epoch_row(
                epoch=epoch + 1,
                num_epochs=num_epochs,
                train_loss=train_loss_epoch,
                train_acc=train_acc_epoch,
                valid_acc=valid_acc_epoch,
                train_f1=train_f1_epoch,
                valid_f1=valid_f1_epoch,
                best_valid_f1=best_valid_f1,
                status=checkpoint_status,
            )
        )

        model.train()

    tqdm.write("")
    tqdm.write(f"Best checkpoint: {best_model_path}")

    return history


def train_baseline_model(model, dataloaders, device, config):
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    return train_CNN(
        model=model,
        num_epochs=config["epochs"],
        dataloaders=dataloaders,
        optimizer=optimizer,
        loss_func=criterion,
        device=device,
        val_every_steps=config.get("val_every_steps"),
        save_dir=config["checkpoint_dir"],
        stage_name=f"{config['display_name']} from scratch",
    )


def train_transfer_model(model, model_name, dataloaders, device, class_weights, config):
    backbone = get_backbone(model, model_name)
    classifier = get_classifier(model, model_name)

    if config.get("strategy") == "two_phase_no_freeze":
        for parameter in model.parameters():
            parameter.requires_grad = True

        phase1_optimizer = torch.optim.AdamW(
            [
                {"params": backbone.parameters(), "lr": config["phase1_backbone_lr"]},
                {"params": classifier.parameters(), "lr": config["phase1_classifier_lr"]},
            ],
            weight_decay=config.get("weight_decay", 0.0),
        )
        phase1_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        phase1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            phase1_optimizer,
            T_max=config["phase1_epochs"],
            eta_min=config.get("phase1_eta_min", config.get("eta_min", 1e-6)),
        )

        history = train_CNN(
            model=model,
            num_epochs=config["phase1_epochs"],
            dataloaders=dataloaders,
            optimizer=phase1_optimizer,
            loss_func=phase1_criterion,
            device=device,
            val_every_steps=config.get("val_every_steps"),
            save_dir=config["phase1_dir"],
            scheduler=phase1_scheduler,
            stage_name=f"{config['display_name']} fine-tuning phase 1",
        )

        best_model_path = os.path.join(config["phase1_dir"], "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            tqdm.write(f"loaded best phase 1 checkpoint: {best_model_path}")

        phase2_optimizer = torch.optim.AdamW(
            [
                {"params": backbone.parameters(), "lr": config["phase2_backbone_lr"]},
                {"params": classifier.parameters(), "lr": config["phase2_classifier_lr"]},
            ],
            weight_decay=config.get("weight_decay", 0.0),
        )
        phase2_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            phase2_optimizer,
            T_max=config["phase2_epochs"],
            eta_min=config.get("phase2_eta_min", config.get("eta_min", 1e-6)),
        )

        return train_CNN(
            model=model,
            num_epochs=config["phase2_epochs"],
            dataloaders=dataloaders,
            optimizer=phase2_optimizer,
            loss_func=phase2_criterion,
            device=device,
            val_every_steps=config.get("val_every_steps"),
            save_dir=config["phase2_dir"],
            scheduler=phase2_scheduler,
            history=history,
            stage_name=f"{config['display_name']} fine-tuning phase 2",
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    head_criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=config.get("head_label_smoothing", 0.0),
    )
    head_optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["head_lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    history = train_CNN(
        model=model,
        num_epochs=config["head_epochs"],
        dataloaders=dataloaders,
        optimizer=head_optimizer,
        loss_func=head_criterion,
        device=device,
        val_every_steps=config.get("val_every_steps"),
        save_dir=config["stage1_dir"],
        stage_name=f"{config['display_name']} classifier head",
    )

    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": config["backbone_lr"]},
            {"params": classifier.parameters(), "lr": config["classifier_lr"]},
        ],
        weight_decay=config.get("weight_decay", 0.0),
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.get("fine_tune_label_smoothing", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["fine_tune_epochs"],
        eta_min=config.get("eta_min", 1e-6),
    )

    return train_CNN(
        model=model,
        num_epochs=config["fine_tune_epochs"],
        dataloaders=dataloaders,
        optimizer=optimizer,
        loss_func=criterion,
        device=device,
        val_every_steps=config.get("val_every_steps"),
        save_dir=config["stage2_dir"],
        scheduler=scheduler,
        history=history,
        stage_name=f"{config['display_name']} fine-tuning",
    )


def summarize_history(model_name, history, parameter_count):
    return {
        "model": model_name,
        "parameters": int(parameter_count),
        "best_valid_macro_f1": float(_best_history_value(history, "valid_f1")),
        "best_valid_accuracy": float(_best_history_value(history, "valid_acc")),
        "last_train_loss": float(history["train_loss_epoch"][-1]) if history["train_loss_epoch"] else None,
        "epochs_completed": len(history["train_loss_epoch"]),
    }


def plot_history(history):
    if "train_acc_step" in history and len(history["train_acc_step"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_acc_step"], label="train_acc_step")
        plt.plot(history["valid_acc_step"], label="valid_acc_step")
        plt.xlabel("Validation checkpoint")
        plt.ylabel("Accuracy")
        plt.title("Step Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "train_f1_step" in history and len(history["train_f1_step"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_f1_step"], label="train_f1_step")
        plt.plot(history["valid_f1_step"], label="valid_f1_step")
        plt.xlabel("Validation checkpoint")
        plt.ylabel("Macro F1")
        plt.title("Step Macro F1")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "train_acc_epoch" in history and len(history["train_acc_epoch"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_acc_epoch"], label="train_acc_epoch")
        plt.plot(history["valid_acc_epoch"], label="valid_acc_epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Epoch Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "train_f1_epoch" in history and len(history["train_f1_epoch"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_f1_epoch"], label="train_f1_epoch")
        plt.plot(history["valid_f1_epoch"], label="valid_f1_epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Epoch Macro F1")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "train_loss_step" in history and len(history["train_loss_step"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss_step"], label="train_loss_step")
        plt.xlabel("Train step")
        plt.ylabel("Loss")
        plt.title("Step Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    if "train_loss_epoch" in history and len(history["train_loss_epoch"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss_epoch"], label="train_loss_epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


def predict(model, loader, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    all_predictions = torch.tensor([], device=device, dtype=torch.int)
    print("Test mode...")

    for inputs in tqdm(loader, desc="Predicting"):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.argmax(-1).int()
            all_predictions = torch.cat((all_predictions, predictions), 0)

    return all_predictions.cpu()
