import torch
import torch.optim as optim
from models.vit import get_vit
from models.mae import MAE
from dataset import get_dataloader
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "./data/unlabeled"
with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]
epochs = config["epochs"]
lr = config["learning_rate"]

train_loader, _ = get_dataloader(data_dir, batch_size)

encoder = get_vit(pretrained=False)
model = MAE(encoder).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
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