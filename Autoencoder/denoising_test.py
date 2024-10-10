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


# training 
def test():
    data_path = '../Data'
    train_loader, val_loader, test_loader = utils.get_data(data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # train된 모델 불러오기
    model = AE_model.AutoEncoder().to(device)
    best_model_path = './best_noise_autoencoder_model.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    loss_fun = nn.MSELoss()

    model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            noise = nn.init.normal_(torch.FloatTensor(inputs.size(0), 1, 28, 28), 0, 0.1)
            noise = noise.to(device)
            inputs = inputs.to(device)
            noise_inputs = inputs + noise

            outputs, encoded = model(noise_inputs)
            test_loss = loss_fun(inputs, outputs)

            running_loss += test_loss.item() * inputs.size(0)
        
        test_loss = running_loss / len(test_loader.dataset)
        print(f"test 손실 결과: {test_loss}")

    out_img = torch.squeeze(outputs.cpu().data)
    print(out_img.size())

    fig, axes = plt.subplots(8, 2, figsize=(8, 10))  # 5개의 이미지, 각 이미지 쌍을 저장
    for i in range(8):
        # noise 원본 이미지 (왼쪽)
        axes[i, 0].imshow(torch.squeeze(noise_inputs[i]).cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # 오토인코더 출력 이미지 (오른쪽)
        axes[i, 1].imshow(out_img[i].numpy(), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('compare_noise_test_img.png')
    print("비교 결과 이미지가 'compare_noise_test_img.png'로 저장되었습니다.")


test()
