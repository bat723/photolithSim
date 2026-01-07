"""Train the SIMPLE autoencoder."""
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')

from src.detection.autoencoder_simple import SimpleAutoencoder
from src.detection.dataset import create_dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, _ in pbar:
        images = images.to(device)
        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    with open('configs/train_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use SIMPLE model with tiny latent dimension
    print("\nInitializing SIMPLE model...")
    model = SimpleAutoencoder(latent_dim=16).to(device)  # Much smaller!
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nLoading training data...")
    train_loader = create_dataloaders(
        'data/processed_images/train',
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers']
    )
    print(f"Training samples: {len(train_loader.dataset)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    print("\nStarting training...")
    best_loss = float('inf')

    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Average loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/autoencoder_simple_best.pth')
            print("✓ Saved new best model")

    torch.save(model.state_dict(), 'models/autoencoder_simple_final.pth')
    print(f"\n✓ Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
