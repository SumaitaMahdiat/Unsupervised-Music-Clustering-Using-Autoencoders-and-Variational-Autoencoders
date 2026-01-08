import torch
from torch import nn
import numpy as np

# Paths
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"

# Load data
X_train = np.load(f"{output_folder}/X_train.npy")

# Parameters
input_dim = X_train.shape[1]   # from Script 1
latent_dim = 32                # size of latent space (tuneable)
condition_dim = 1              # genre label

# CVAE Architecture
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, 256)
        self.fc2_mu = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
        
        self.relu = nn.ReLU()
    
    def encode(self, x, c):
        # x: input, c: condition (genre)
        h = self.relu(self.fc1(torch.cat([x, c], dim=1)))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        h = self.relu(self.fc3(torch.cat([z, c], dim=1)))
        return self.fc4(h)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CVAE(input_dim, latent_dim, condition_dim).to(device)
print(model)

# Save model
torch.save(model.state_dict(), f"{output_folder}/cvae_model.pth")
