import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DataSetup import create_dataloaders
from model import create_model
from engine import train_model
from evaluate import evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)

    data_dir = 'data'
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, dataset_sizes, class_names = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=2,
        n_segments=100, compactness=10
    )

    print(f"Classes: {class_names}")
    print(f"Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    print(f"Device: {device}\n")

    model = create_model(num_classes=len(class_names), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    trained_model = train_model(
        model, dataloaders, dataset_sizes,
        criterion, optimizer, scheduler, device, num_epochs
    )

    evaluate_model(trained_model, dataloaders['val'], device, class_names)

    torch.save(trained_model.state_dict(), 'best_model.pth')
    print("Model saved to best_model.pth")


if __name__ == '__main__':
    main()