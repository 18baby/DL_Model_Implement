import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Basic_CNN(nn.Module):
    """
    기본 CNN 모델
    입력: 1x28x28 MNIST 이미지
    Layer 1: 32개 3x3 filter(Stride=1), BN, ReLU, 2x2 MaxPooling(Stride=2)
    Layer 2: 64개 3x3 filter(Stride=1), BN, ReLU, 2x2 MaxPooling(Stride=1)
    Layer 3: FC 128, ReLU 
    Layer 4: FC 10 -> 일반적으로 softmax는 loss 함수(cross entropy)에서 규정 
    """
    def __init__(self):
        super(Basic_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),  # 32*26*26
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 32*13*13
        )   
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1),   # 64*11*11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)   # 64*10*10 
        )   
        self.layer3 = nn.Sequential(
            nn.Linear(64*10*10, 128),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(128, 10)

        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        out = self.layer4(x)

        return out



