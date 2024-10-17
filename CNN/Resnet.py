import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
The overall structure of ResNet is like below.

* input(channel:3) -> (conv 3x3) -> (bn) -> (relu) -> output(channel:16)
* n Residual blocks: (16 channels -> 16 channels)
* n Residual blocks: (16 channels -> 32 channels)
* n Residual blocks: (32 channels -> 64 channels)
* global average pooling + fully connected layer
"""

# Residual block 단위
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) # (N-3+2)+1 -> 차원 유지
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.down_sample = down_sample

        self.stride = stride
        self.in_channel = in_channels
        self.out_channel = out_channels

    # out_channel이 in_channel 보다 큰 경우에 입력 크기와 맞추기 위해서 max pooling을 진행
    def down_sampling(self, x):
        # self.out_channel - self.in_channel: 체널 차원에 패딩을 추가 -> 주워진 텐서에 패딩을 추가해 차원을 확장
        # 3D 텐서에 대해 (left, right, top, bottom, front, back) 순서로 지정
        out = F.pad(x, (0, 0, 0, 0, 0, self.out_channel - self.in_channel))
        out = nn.MaxPool2d(kernel_size=2, stride=self.stride)(out)
        return out

    def forward(self, x):
        short_cut = x      # residual 함
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample == True:
            short_cut = self.down_sampling(x)

        out += short_cut
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10) -> None:
        super(ResNet, self).__init__()
        self.num_layer = num_layers

        # input(channel:3) -> (conv 3x3) -> (bn) -> (relu) -> output(channel:16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # feature map size = 16x32x32
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.get_layer(block, in_channels=16, out_channels=16, stride=1) # feature map size = 16x32x32

        self.layer2 = self.get_layer(block, in_channels=16, out_channels=32, stride=2) # feature map size = 32x16x16

        self.layer3 = self.get_layer(block, in_channels=32, out_channels=64, stride=2) # feature map size = 64x8x8

        # output 계산
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def get_layer(self, block, in_channels, out_channels, stride):
        # 초기에 차원이 줄어들때만 다운셈플링 진행
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        # layer들 생성
        layer_list = nn.ModuleList([block(in_channels, out_channels, stride, down_sample)])

        for _ in range(self.num_layer-1):
            layer_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layer_list)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)

        return out