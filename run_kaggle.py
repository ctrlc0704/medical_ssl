import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import timm

# ======================
# CONFIG
# ======================

DATA_ROOT = "/kaggle/input/CheXpert-v1.0-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS_PRETRAIN = 3
EPOCHS_FINETUNE = 3
MASK_RATIO = 0.75
MODEL_NAME = "vit_base_patch16_224"

SAVE_PATH = "/kaggle/working"

print("Using device:", DEVICE)

# ======================
# DATASET
# ======================

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.labels = [
            'Atelectasis','Cardiomegaly',
            'Consolidation','Edema',
            'Pleural Effusion'
        ]

        self.df[self.labels] = self.df[self.labels].fillna(0)
        self.df[self.labels] = self.df[self.labels].replace(-1, 0)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.iloc[idx, 0]

        # Remove prefix from CSV
        img_rel_path = img_rel_path.replace("CheXpert-v1.0-small/", "")

        img_path = os.path.join(self.root_dir, img_rel_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            self.df.iloc[idx][self.labels].values.astype(float)
        )

        return image, label

# ======================
# MAE MODEL
# ======================

class MAE(nn.Module):
    def __init__(self, encoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        embed_dim = encoder.num_features

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3*224*224)
        )

    def random_mask(self, x):
        mask = torch.rand_like(x[:, :1])
        mask = (mask > self.mask_ratio).float()
        return x * mask

    def forward(self, x):
        x_masked = self.random_mask(x)
        latent = self.encoder(x_masked)
        recon = self.decoder(latent)
        recon = recon.view(x.size())
        loss = nn.functional.mse_loss(recon, x)
        return loss

# ======================
# PRETRAIN
# ======================

print("Loading dataset...")

train_dataset = CheXpertDataset(
    os.path.join(DATA_ROOT,"train.csv"),
    DATA_ROOT,
    train=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

encoder = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=0
)

mae_model = MAE(encoder, MASK_RATIO).to(DEVICE)
optimizer = optim.AdamW(mae_model.parameters(), lr=LR)

print("Starting MAE pretraining...")

for epoch in range(EPOCHS_PRETRAIN):
    mae_model.train()
    total_loss = 0

    for images, _ in tqdm(train_loader):
        images = images.to(DEVICE)

        loss = mae_model(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Pretrain] Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

torch.save(mae_model.encoder.state_dict(),
           os.path.join(SAVE_PATH,"mae_encoder.pth"))

print("Saved encoder.")

# ======================
# FINETUNE
# ======================

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

print("Starting finetuning...")

val_dataset = CheXpertDataset(
    os.path.join(DATA_ROOT,"valid.csv"),
    DATA_ROOT,
    train=False
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

backbone = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=0
)

backbone.load_state_dict(
    torch.load(os.path.join(SAVE_PATH,"mae_encoder.pth"))
)

model = Classifier(backbone, 5).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

best_auc = 0

for epoch in range(EPOCHS_FINETUNE):
    model.train()

    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            y_true.append(labels.numpy())
            y_pred.append(torch.sigmoid(outputs).cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    auc = roc_auc_score(y_true, y_pred, average="macro")
    print(f"[Finetune] Epoch {epoch+1} AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_PATH,"best_model.pth"))
        print("Saved best model.")

print("Training completed.")
print("Best AUC:", best_auc)
