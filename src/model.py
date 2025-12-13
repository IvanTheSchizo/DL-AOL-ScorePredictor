import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder: Compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # Compressed Representation
        )
        
        # Decoder: Expand
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim) # Back to original size
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent