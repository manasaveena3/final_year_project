# model_code.py
import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel


class HybridViTResNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTResNet, self).__init__()
        
        # ResNet Feature Extractor
        self.resnet = resnet
        self.resnet_fc = nn.Linear(2048, 512)  # ResNet output to 512
        
        # Vision Transformer Feature Extractor
        self.vit = vit
        self.vit_fc = nn.Linear(768, 512)  # ViT output to 512

        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 256),  # Merge ResNet & ViT features
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, x):
        resnet_features = self.resnet(x).view(x.size(0), -1)
        resnet_features = self.resnet_fc(resnet_features)

        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        vit_features = self.vit_fc(vit_features)

        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        output = self.fc(combined_features)

        return output