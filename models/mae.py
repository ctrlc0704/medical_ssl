import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.num_features

        self.decoder = nn.Linear(embed_dim, 3 * 224 * 224)

    def forward(self, x):
        latent = self.encoder.forward_features(x)
        recon = self.decoder(latent[:, 0])
        recon = recon.view(x.size())
        loss = F.mse_loss(recon, x)
        return loss