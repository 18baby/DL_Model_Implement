import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib as mpl
import VAE_model
import utils


# training 
def test():
    data_path = '../Data'
    train_loader, val_loader, test_loader = utils.get_data(data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # train된 모델 불러오기
    model = VAE_model.VariationalAutoEncoder().to(device)
    best_model_path = './best_VAE_model.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    loss_fun = nn.MSELoss()

    model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs, mu, log_var = model(inputs)
            test_loss = utils.loss_fn(inputs, outputs, mu, log_var)

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

    return test_loader, model, device


def main():
    test_loader, model, device = test()
    
    # 전체 데이터의 결과 확인
    test_data = test_loader.dataset
    all_labels = np.array([label for _, label in test_data])
    all_imgs = [data for data, _ in test_data]
    all_imgs = torch.stack(all_imgs).to(device)    # 텐서로 변환
    outputs, mu, log_var = model(all_imgs)

    encoded = mu.cpu().detach().numpy()
    print(f"전체 test 데이터: {encoded.shape}")

    # 각 라벨별 z벡터의 평균값 계산
    mean_encoded = []
    for i in range(10):
        mean_encoded.append(encoded[np.where(all_labels == i)[0]].mean(axis=0))

    # 임의의 두 라벨을 변형해가기
    selected_class = [1, 7]
    samples = []
    with torch.no_grad():
        for idx, coef in enumerate(np.linspace(0, 1, 10)):
            interpolated = coef * mean_encoded[selected_class[0]] + (1.-coef) * mean_encoded[selected_class[1]]
            samples.append(interpolated)
        samples = np.stack(samples)
        z = torch.tensor(samples).to(device).float()

        generated = model.decoder(z).to(device)

    generated = generated.view(10, 1, 28, 28)
    img = make_grid(generated, nrow=10)
    npimg = img.cpu().numpy()
    plt.clf()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig(f'vae_change{selected_class[0]}to{selected_class[1]}.png')

if __name__ == "__main__":
    main()
