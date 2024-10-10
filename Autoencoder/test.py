import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import AE_model
import time
import copy

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


# training 
def test():
    data_path = '../Data'
    train_loader, val_loader, test_loader = get_data(data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # train된 모델 불러오기
    model = AE_model.AutoEncoder().to(device)
    best_model_path = './best_autoencoder_model.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    loss_fun = nn.MSELoss()

    model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs, encoded = model(inputs)
            test_loss = loss_fun(inputs, outputs)

            running_loss += test_loss.item() * inputs.size(0)
        
        test_loss = running_loss / len(test_loader.dataset)
        print(f"test 손실 결과: {test_loss}")

    out_img = torch.squeeze(outputs.cpu().data)
    print(out_img.size())

    fig, axes = plt.subplots(8, 2, figsize=(8, 10))  # 5개의 이미지, 각 이미지 쌍을 저장
    for i in range(8):
        # 원본 이미지 (왼쪽)
        axes[i, 0].imshow(torch.squeeze(inputs[i]).cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # 오토인코더 출력 이미지 (오른쪽)
        axes[i, 1].imshow(out_img[i].numpy(), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('compare_test_img.png')
    print("비교 결과 이미지가 'compare_test_img.png'로 저장되었습니다.")


test()
