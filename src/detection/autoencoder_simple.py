"""Ultra-simple autoencoder with very limited capacity."""
import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """Tiny autoencoder - forces strong compression."""

    def __init__(self, latent_dim=16):  # MUCH smaller!
        super().__init__()

        # Encoder: 256x256 → 16-dim bottleneck (TINY!)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 256→128
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 128→64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # 32→16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent_dim)  # EXTREME compression!
        )

        self.decoder_fc = nn.Linear(latent_dim, 32 * 16 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),   # 128→256
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder_fc(z)
        z = z.view(-1, 32, 16, 16)
        return self.decoder(z)
