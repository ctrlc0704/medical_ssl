import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from models.vit import get_vit
from dataset import get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True)
parser.add_argument("--val_csv", type=str, required=True)
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--mode", type=str, default="scratch")
args = parser.parse_args()

train_loader = get_loader(args.train_csv, args.root_dir)
val_loader = get_loader(args.val_csv, args.root_dir, train=False)

num_classes = 5

if args.mode == "scratch":
    model = get_vit(pretrained=False)
elif args.mode == "imagenet":
    model = get_vit(pretrained=True)
elif args.mode == "medical_ssl":
    model = get_vit(pretrained=False)
    model.load_state_dict(torch.load("mae_encoder.pth"))

model.head = nn.Linear(model.num_features, num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        y_true.append(labels.numpy())
        y_pred.append(outputs.cpu().numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

auc = roc_auc_score(y_true, y_pred, average="macro")
print("Mode:", args.mode)
print("Macro AUC:", auc)