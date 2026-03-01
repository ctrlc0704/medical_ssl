import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

def get_dataloader(data_dir, batch_size=32, train=True):
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(train)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )
    return loader, len(dataset.classes)