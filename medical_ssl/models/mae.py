import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio

        embed_dim = encoder.num_features

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 3 * 224 * 224)
        )

    def random_mask(self, x):
        B, C, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device)
        mask = (mask > self.mask_ratio).float()
        return x * mask

    def forward(self, x):
        x_masked = self.random_mask(x)
        latent = self.encoder(x_masked)
        recon = self.decoder(latent)
        recon = recon.view(x.size())
        loss = F.mse_loss(recon, x)
        return loss