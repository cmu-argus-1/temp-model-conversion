
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


LD_MODEL_SUF = ".pth"
NUM_CLASS = 16



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


class RegionClassifier:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClassifierEfficient(NUM_CLASS).to(self.device)

        # Load Custom model weights
        model_weights_path = os.path.join("../model_effnet_0.997_acc" + LD_MODEL_SUF)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

        # Define the preprocessing
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )



if __name__ == "__main__":
    rc = RegionClassifier()

    rc_scripted = torch.jit.script(rc.model) 
    rc_scripted.save("model_scripted.pt")