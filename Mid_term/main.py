import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy
import random


def load_data():
    # 데이터 변환 (이미지를 텐서로 변환)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    # FashionMNIST 데이터셋 불러오기 (train, test 나눔)
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # train 데이터셋을 train과 validation으로 나누기
    train_size = int(0.8 * len(train_dataset))  # 80%는 train
    val_size = len(train_dataset) - train_size  # 나머지 20%는 validation
    check_train_dataset, check_val_dataset = random_split(train_dataset, [train_size, val_size])

    # DataLoader로 데이터셋을 배치(batch)로 나눠서 로딩
    check_train_loader = DataLoader(dataset=check_train_dataset, batch_size=128, shuffle=True)
    check_val_loader = DataLoader(dataset=check_val_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    return check_train_loader, check_val_loader, train_loader, test_loader

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # input(channel:1) -> (conv 3x3) -> (bn) -> (relu) -> output(channel:16)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # feature map size = 16x28x28
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.get_layer(block, in_channels=16, out_channels=16, stride=1) # feature map size = 16x28x28

        self.layer2 = self.get_layer(block, in_channels=16, out_channels=32, stride=2) # feature map size = 32x14x14

        self.layer3 = self.get_layer(block, in_channels=32, out_channels=64, stride=2) # feature map size = 64x7x7

        # output 계산
        self.avg_pool = nn.AvgPool2d(4, stride=1)
        self.fc_out = nn.Linear(64*4*4, num_classes)

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

def train_val_check(model, train_loader, val_loader, device, epochs=10):
    # 손실함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_his = []
    val_loss_his = []
    train_acc = []
    val_acc = []

    # 최고 성능 모델 저장
    best_model = copy.deepcopy(model.state_dict())
    best_model_loss = 1e+8
    best_epoch = 0

    model.to(device)

    for epoch in range(epochs):

        for state in ['train', 'val']:
            running_loss = 0.0
            running_acc = 0.0  # 배치별 누적 정확도 초기화
            correct = 0  # 정확하게 예측한 샘플 수
            total = 0    # 전체 샘플 수

            if state == 'train':
                data_loader = train_loader
                model.train()
            else:
                data_loader = val_loader
                model.eval()

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(state=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, predict = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += torch.sum(predict==labels).item()
                
                    if state == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_acc = correct / total
            epoch_loss = running_loss / len(data_loader.dataset)   # epoch별 평균 손실 계산

            print(f'Epoch [{epoch+1}/{epochs}], {state} Loss: {epoch_loss:.4f}, {state} Accuracy: {100 * epoch_acc:.2f}%')

            if state == 'train':
                train_loss_his.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif state == 'val':
                val_loss_his.append(epoch_loss)
                val_acc.append(epoch_acc)
            if (state=='val') and (epoch_loss < best_model_loss):
                best_model_loss = epoch_loss
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())   # 최고 loss의 파라미터 저장
        print()
    print(f'Best val Loss: {best_model_loss:4f}, best epoch: {best_epoch}')

    model.load_state_dict(best_model_wts)

    return model

def train_test(model, train_loader, test_loader, device, epochs=10):
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 더 자긍ㄴ 학습률로 업데이트

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
    all_logits = []     # 테스트 전체 결과 저장
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_logits.extend(outputs.cpu().numpy())
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # test 결과 저장
    np.save('20243679.npy', np.array(all_logits))
    


def main():
    set_seed()

    check_train_loader, check_val_loader, train_loader, test_loader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 확인
    # for images, labels in train_loader:
    #     print(f'Image batch shape: {images.shape}')
    #     print(f'Label batch shape: {labels.shape}')
    #     break

    block = ResidualBlock
    model = ResNet(10, block)

    # train, val로 성능 확인
    model = train_val_check(model, check_train_loader, check_val_loader, device, 10)

    print("Train with full training data (train_loader)...")
    # 전체데이터로 학습 및 test 진행
    train_test(model, train_loader, test_loader, device, 5)




if __name__ == "__main__":
    main()