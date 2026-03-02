import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir="", transform=None):
        self.df = pd.read_csv(csv_file)

        self.labels = [
            'Atelectasis','Cardiomegaly',
            'Consolidation','Edema',
            'Pleural Effusion'
        ]

        self.df = self.df[['Path'] + self.labels]
        self.df[self.labels] = self.df[self.labels].fillna(0)
        self.df[self.labels] = self.df[self.labels].replace(-1, 0)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.df.iloc[idx, 1:].values).float()

        return image, label


def get_loader(csv_path, root_dir="", batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = CheXpertDataset(csv_path, root_dir, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )