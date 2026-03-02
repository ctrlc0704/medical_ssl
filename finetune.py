import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vit import get_vit
from datasets.chexpert import CheXpertDataset, get_transform
from engine.trainer import train_epoch
from engine.evaluator import evaluate

def run_finetune(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CheXpertDataset(cfg["train_csv"], cfg["data_root"], get_transform())
    val_dataset = CheXpertDataset(cfg["val_csv"], cfg["data_root"], get_transform())

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    model = get_vit(pretrained=False)
    model.load_state_dict(torch.load("mae_encoder.pth"))
    model.head = nn.Linear(model.num_features, cfg["num_classes"])
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch} Loss {loss:.4f} AUC {auc:.4f}")