"""Convolutional Auto-Encoder for anomaly detection."""
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """CNN Auto-Encoder: 256x256 → 64 latent → 256x256"""

    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder: compress 256x256 → 64-dim bottleneck
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 256→128
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 128→64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64→32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, latent_dim)
        )

        self.decoder_fc = nn.Linear(latent_dim, 32 * 32 * 32)

        # Decoder: reconstruct 64-dim → 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),   # 128→256
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder_fc(z)
        z = z.view(-1, 32, 32, 32)
        reconstructed = self.decoder(z)
        return reconstructed

    def get_reconstruction_error(self, x):
        """Calculate per-image reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
        return error
