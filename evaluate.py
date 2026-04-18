import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=class_names)
    matrix = confusion_matrix(all_labels, all_preds)

    print("Classification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(matrix)

    return all_labels, all_preds