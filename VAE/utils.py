import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 데이터 불러오기
def get_data(data_path, batch_size=64):
    # MNIST training set
    train_dataset = dset.MNIST(
        root=data_path, 
        train=True, 
        transform=transforms.ToTensor(), 
        download=False
    )

    # MNIST test set
    test_dataset = dset.MNIST(
        root=data_path, 
        train=False, 
        transform=transforms.ToTensor(), 
        download=False
    )

    # train, val 분리
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # DataLoader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('데이터셋 불러오기 성공')

    return train_loader, val_loader, test_loader

# VAE 손실함수 정의
def loss_fn(x, z, mu, log_var):
    BCE = torch.nn.BCELoss(reduction='sum')
    bce_loss = BCE(z.view(-1, 784), x.view(-1, 784))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())    # KL loss 계산식
    return bce_loss + kl_loss