import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ClassifierEfficient(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierEfficient, self).__init__()
        # Using new weights system
        # This uses the most up-to-date weights
        weights = EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = efficientnet_b0(weights=weights)
        for param in self.efficientnet.features[:3].parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x
