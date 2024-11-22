import os
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Function to set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to prepare test data
def prepare_test_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to calculate accuracy
def calculate_accuracy(logits, targets):
    pred = np.argmax(logits, axis=1)
    accuracy = 100. * np.sum(pred == targets.numpy()) / len(targets)
    return round(accuracy, 2)

# Main function to evaluate logits
def evaluate(logits_filename):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = prepare_test_data()
    print(f'Test dataset size: {len(test_loader.dataset)}')

    try:
        # Ensure file name is valid
        assert re.match(r"\d{8}\.npy", logits_filename), "File name must be an 8-digit student ID followed by '.npy'."

        # Load logits
        logits = np.load(logits_filename)
        assert logits.shape == (len(test_loader.dataset), 10), f"Logits shape mismatch: expected ({len(test_loader.dataset)}, 10)."

        # Calculate accuracy
        targets = torch.cat([target for _, target in test_loader]).cpu()
        accuracy = calculate_accuracy(logits, targets)

    except AssertionError as e:
        accuracy = 0.0
        print(f"Evaluation failed: {e}")

    print(f'{logits_filename[:-4]} - Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate("20243679.npy")
