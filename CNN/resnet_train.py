import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim
import Resnet

# 데이터셋을 저장할 경로 설정
data_dir = '../Data/Data'

# 데이터 로더를 위한 전처리 및 CIFAR-10 데이터 불러오기
def load_data():
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 데이터셋 다운로드 및 로딩
    cifar_train = CIFAR10(root=data_dir, train=True, download=False, transform=transforms_train)
    cifar_test = CIFAR10(root=data_dir, train=False, download=False, transform=transforms_test)

    # DataLoader로 학습 및 테스트 데이터 불러오기
    train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=4)
    print("CIFIAR10 데이터 불러오기 완료")
    return train_loader, test_loader

# 간단한 학습 및 테스트 루틴
def train_model(model, train_loader, test_loader, device, epochs=10):
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%')

        # 테스트 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

# main 함수
def main():
    # CUDA 또는 CPU 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로딩
    train_loader, test_loader = load_data()

    # 예시로 ResNet18 모델을 사용할 수 있음 (또는 다른 모델 정의)
    block = Resnet.ResidualBlock
    model = Resnet.ResNet(3, block)  # ResNet18 정의 필요

    # 모델 학습
    train_model(model, train_loader, test_loader, device, epochs=10)

if __name__ == "__main__":
    main()
