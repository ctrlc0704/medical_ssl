import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            y_true.append(labels.numpy())
            y_pred.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    auc = roc_auc_score(y_true, y_pred, average="macro")
    return auc