import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchmetrics
import matplotlib as mpl
import basic_cnn as cnn
import utils


# training 
def test():
    data_path = '../Data'
    train_loader, val_loader, test_loader = utils.get_data(data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # train된 모델 불러오기
    model = cnn.Basic_CNN().to(device)
    best_model_path = './best_basicCNN_model.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    loss_fun = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

    model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        correct_predictions = 0  # 정확하게 예측한 총 샘플 수
        total_predictions = 0     # 전체 샘플 수

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            test_loss = loss_fun(outputs, labels)
            running_loss += test_loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

        
        test_loss = running_loss / len(test_loader.dataset)
        test_acc = correct_predictions / total_predictions
        print(f"test 손실 결과: {test_loss:.4f}, Accuracy: {test_acc}")


test()
