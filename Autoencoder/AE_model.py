import numpy as np
import torch
import torch.nn as nn


# 모델 구조: 784(28*28) -> 512 -> 128 -> 32 -> 128 -> 512 -> 784(28*28)
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, input):
        # batch 단위 처리
        batch_size = input.size(0)
        x = input.view(-1, 28*28)   # (batch_size, 784) tensor로 만들기

        z = self.encoder(x)
        output = self.decoder(z).view(batch_size, 1, 28, 28)   # 원래 이미지 차원 복원

        return output, z


