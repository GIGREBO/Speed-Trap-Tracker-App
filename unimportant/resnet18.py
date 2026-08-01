import torch.nn as nn 
from torchvision import models
from torchvision.models import ResNet18_Weights


model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x 


model = MLC(num_classes=10)
print(model.resnet)


for name, param in model.resnet.named_parameters():
    if param.requires_grad:
        print(name, param.data)
