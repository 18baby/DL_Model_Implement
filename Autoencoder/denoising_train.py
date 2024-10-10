import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib as mpl
import AE_model
import utils
import time
import copy


# training 
def training(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, device):
    since = time.time()

    train_loss_his = []
    val_loss_his = []

    best_model_wts = copy.deepcopy(model.state_dict())   # 최적 모델의 파라미터 저장
    best_model_loss = 1e+8

    # epoch 반복 실행
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        running_loss = 0.0

        for state in ['train', 'val']:
            if state == 'train':
                dataloader = train_dataloader
            else:
                dataloader = val_dataloader

            # 배치별 학습 시작
            for inputs, labels in dataloader:
                inputs = inputs.to(device)

                # noise를 추가
                noise = torch.zeros(inputs.size(0), 1, 28, 28)
                nn.init.normal_(noise, 0, 0.1)    # 정규분포로 noise 생성
                noise = noise.to(device)
                noise_input = inputs + noise      # noise가 결합된 input 생성

                optimizer.zero_grad()   # 가중치 초기화

                if state == 'train':
                    model.train()
                else:
                    model.eval()

                with torch.set_grad_enabled(state=='train'):
                    outputs, encoded = model(noise_input)
                    loss = criterion(outputs, inputs)
                    
                    if state == 'train':
                        loss.backward()    # backpropagation 진행
                        optimizer.step()   # optimizer 업데이트
                
                running_loss += loss.item() * inputs.size(0)   # loss.item = loss값 가져오기(배치 평균 손실)

            epoch_loss = running_loss / len(dataloader.dataset)   # epoch별 평균 손실 계산
            print('{} Loss: {:.4f}'.format(state, epoch_loss))

            if state == 'train':
                train_loss_his.append(epoch_loss)
            elif state == 'val':
                val_loss_his.append(epoch_loss)
            if (state=='val') and (epoch_loss < best_model_loss):
                best_model_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())   # 최고 loss의 파라미터 저장
        print()
    
    total_time = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val Loss: {:4f}'.format(best_model_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_loss_his, val_loss_his

def main():
    data_path = '../Data'
    train_loader, val_loader, test_loader = utils.get_data(data_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    learning_rate = 0.0002
    num_epochs = 10
    model = AE_model.AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_model, train_loss_his, val_loss_his = training(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)

    model_save_path = './best_noise_autoencoder_model.pth'
    torch.save(best_model.state_dict(), model_save_path)
    print(f'최적 모델이 {model_save_path}에 저장되었습니다.')

    # 결과 시각화
    mpl.use('Agg')
    plt.plot(train_loss_his, label='train')
    plt.plot(val_loss_his, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # 그래프를 파일로 저장 (예: 'training_loss.png')
    plt.savefig('noise_training_loss.png')
    print("학습 손실 그래프가 'noise_training_loss.png'에 저장되었습니다.")


if __name__ == "__main__":
    main()