import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


def create_efficientnet_v2_s(num_classes, dropout=0.3):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model
