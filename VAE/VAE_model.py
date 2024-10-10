import numpy as np
import torch
import torch.nn as nn


# 모델 구조: 784(28*28) -> 512 -> 128 -> 32 -> 128 -> 512 -> 784(28*28)
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh()
        )

        # latent space의 평균, 분산 추정 -> 직접적으로 값을 구하는게 아니라 통계적으로 추정하는 것!
        self.fc_mu = nn.Linear(256, 10)
        self.fc_var = nn.Linear(256, 10)

        self.decoder = nn.Sequential(
            nn.Linear(10, 256),
            nn.Tanh(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    # latent space의 평균, log 분산 계산
    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    # latent vector Z 계산(샘플링을 미분가능하게 설정)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)   # 표준 편차 계산
        eps = torch.randn_like(std)    # 정규분포 오차 추가 -> 함수: 주어진 tensor와 같은 크기의 표준 정규분포 값을 가지는 텐서 리턴
        return mu + std*eps

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, input):
        # batch 단위 처리
        batch_size = input.size(0)
        x = input.view(-1, 28*28)   # (batch_size, 784) tensor로 만들기

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z).view(batch_size, 1, 28, 28)   # 원래 이미지 차원 복원

        return output, mu, log_var


