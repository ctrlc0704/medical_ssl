import torch
import torch.nn as nn
import torch.optim as optim
from models.vit import get_vit
from dataset import get_dataloader
from utils import compute_auc

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "./data/labeled"
batch_size = 32
epochs = 5
lr = 1e-4

train_loader, num_classes = get_dataloader(data_dir, batch_size, train=True)
val_loader, _ = get_dataloader(data_dir, batch_size, train=False)

model = get_vit(pretrained=False)
model.load_state_dict(torch.load("mae_encoder.pth"))
model.head = nn.Linear(model.num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

# evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.softmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("AUC:", compute_auc(y_true, y_pred))