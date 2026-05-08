import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, EfficientNet_V2_S_Weights


class SimpleCNN(nn.Module):
    """Small from-scratch CNN baseline for comparison with transfer learning."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(96 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def create_simple_cnn(num_classes):
    return SimpleCNN(num_classes=num_classes)


def create_densenet121(num_classes, dropout=0.3):
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    return model


def create_efficientnet_v2_s(num_classes, dropout=0.3):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def create_model(model_name, num_classes, dropout=0.3):
    factories = {
        "simple_cnn": create_simple_cnn,
        "densenet121": create_densenet121,
        "efficientnet_v2_s": create_efficientnet_v2_s,
    }

    if model_name not in factories:
        available = ", ".join(sorted(factories))
        raise ValueError(f"Unknown model_name={model_name!r}. Available models: {available}")

    if model_name == "simple_cnn":
        return factories[model_name](num_classes=num_classes)

    return factories[model_name](num_classes=num_classes, dropout=dropout)


def count_parameters(model, trainable_only=False):
    parameters = model.parameters()
    if trainable_only:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
    return sum(parameter.numel() for parameter in parameters)
