import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2


DATA_MODES = ["train", "val", "test"]
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_RESCALE_SIZE = [224, 224]


class SimpsonsDataset(Dataset):
    def __init__(
        self,
        files,
        label_encoder,
        mode,
        rescale_size=None,
        normalize_mean=None,
        normalize_std=None,
    ):
        super().__init__()

        self.files = sorted(files)
        self.mode = mode

        if self.mode not in DATA_MODES:
            raise ValueError(f"{self.mode} is not correct; correct modes: {DATA_MODES}")

        self.label_encoder = label_encoder
        self.len_ = len(self.files)

        rescale_size = rescale_size or DEFAULT_RESCALE_SIZE
        normalize_mean = normalize_mean or DEFAULT_NORMALIZE_MEAN
        normalize_std = normalize_std or DEFAULT_NORMALIZE_STD

        self.train_transform = build_train_transform(
            rescale_size=rescale_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        self.val_transform = build_eval_transform(
            rescale_size=rescale_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        x = self.load_image(self.files[index])
        x = self.transform_images_to_tensors(x)

        if self.mode == "test":
            return x

        path = self.files[index]
        y = self.label_encoder.transform([path.parent.name]).item()
        return x, y

    def load_image(self, file):
        image = Image.open(file).convert("RGB")
        image.load()
        return image

    def transform_images_to_tensors(self, image):
        if self.mode == "train":
            return self.train_transform(image)
        return self.val_transform(image)


def build_train_transform(rescale_size, normalize_mean, normalize_std):
    return v2.Compose([
        v2.Resize([256, 256]),
        v2.RandomResizedCrop(rescale_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.03,
        ),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(normalize_mean, normalize_std),
    ])


def build_eval_transform(rescale_size, normalize_mean, normalize_std):
    return v2.Compose([
        v2.Resize(rescale_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(normalize_mean, normalize_std),
    ])


def collect_image_files(train_dir, test_dir):
    train_val_files = sorted(list(train_dir.rglob("*.jpg")))
    test_files = sorted(list(test_dir.rglob("*.jpg")))
    return train_val_files, test_files


def create_label_encoder(train_val_files):
    label_encoder = LabelEncoder()
    train_val_labels = [path.parent.name for path in train_val_files]
    label_encoder.fit(train_val_labels)
    return label_encoder, train_val_labels


def split_train_val_files(train_val_files, train_val_labels, test_size=0.25, random_state=42):
    return train_test_split(
        train_val_files,
        test_size=test_size,
        stratify=train_val_labels,
        random_state=random_state,
    )


def create_datasets(train_files, val_files, label_encoder):
    train_dataset = SimpsonsDataset(train_files, label_encoder=label_encoder, mode="train")
    val_dataset = SimpsonsDataset(val_files, label_encoder=label_encoder, mode="val")
    return train_dataset, val_dataset


def compute_class_counts(dataset, label_encoder):
    numeric_labels = label_encoder.transform([path.parent.name for path in dataset.files])
    class_counts = np.bincount(numeric_labels, minlength=len(label_encoder.classes_))
    return numeric_labels, class_counts


def create_weighted_dataloaders(
    train_dataset,
    val_dataset,
    label_encoder,
    batch_size=64,
    train_num_workers=2,
    val_num_workers=0,
):
    train_numeric_labels, class_counts = compute_class_counts(train_dataset, label_encoder)
    class_weights_for_sampler = 1.0 / np.sqrt(np.maximum(class_counts, 1))

    sample_weights = torch.DoubleTensor([
        class_weights_for_sampler[label]
        for label in train_numeric_labels
    ])

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
    }

    return loaders, class_counts


def create_test_loader(test_files, label_encoder, batch_size=64):
    test_dataset = SimpsonsDataset(test_files, label_encoder=label_encoder, mode="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return test_dataset, test_loader
