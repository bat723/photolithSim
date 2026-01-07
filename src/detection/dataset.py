"""PyTorch dataset for loading wafer images."""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class WaferDataset(Dataset):
    """Load grayscale images from a folder."""

    def __init__(self, image_folder, transform=None):
        self.image_paths = list(Path(image_folder).glob("*.png"))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, 0


def create_dataloaders(train_folder, batch_size=32, num_workers=2):
    """Helper function to create training dataloader."""
    train_dataset = WaferDataset(train_folder)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader
