import torch
import torch.optim as optim
import argparse
from models.vit import get_vit
from models.mae import MAE
from dataset import get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

train_loader = get_loader(args.csv, args.root_dir)

encoder = get_vit(pretrained=False)
model = MAE(encoder, mask_ratio=0.75).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        loss = model(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

torch.save(model.encoder.state_dict(), "mae_encoder.pth")