import torch
from sklearn.metrics import roc_auc_score

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)