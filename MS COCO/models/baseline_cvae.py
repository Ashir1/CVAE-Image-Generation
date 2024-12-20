import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineConvVAE(nn.Module):
    def __init__(self, latent_dim=128, text_embedding_dim=768):
        super(BaselineConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (B, 512, 4, 4)
            nn.ReLU()
        )

        # Derive latent space using concatenated text, image pair
        self.fc_mu = nn.Linear(512 * 4 * 4 + text_embedding_dim, latent_dim)       
        self.fc_logvar = nn.Linear(512 * 4 * 4 + text_embedding_dim, latent_dim)  

        self.fc_decode = nn.Linear(latent_dim + text_embedding_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (B, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # (B, 3, 64, 64)
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x, captions_emb):
        h = self.encoder(x) 
        h = h.view(h.size(0), -1) 
        h = torch.cat([h, captions_emb], dim=-1)
        mu = self.fc_mu(h)  
        logvar = self.fc_logvar(h)  
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, captions_emb):
        z = torch.cat([z, captions_emb], dim=-1)
        h = self.fc_decode(z)  
        h = h.view(h.size(0), 512, 4, 4) 
        recon_x = self.decoder(h) 
        return recon_x

    def forward(self, x, captions_emb):
        mu, logvar = self.encode(x, captions_emb)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, captions_emb)
        
        # Compute loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')  
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
        loss = recon_loss + kl_loss

        return {
            'recon_img': recon_x,
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar
        }