import torch
import numpy as np
from dataset import MusicDataset
from vae import VAE

# ---------------------------
# CSV and model path
# ---------------------------
csv_file = "C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/train.csv"
model_path = "results/easy/vae_music.pth"

# ---------------------------
# Dataset and device
# ---------------------------
dataset = MusicDataset(csv_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load trained VAE
# ---------------------------
model = VAE(input_dim=dataset.features.shape[1], latent_dim=16).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ---------------------------
# Encode latent features
# ---------------------------
latent_features = []
with torch.no_grad():
    for x in dataset.features:
        x = x.to(device)
        mu, logvar = model.encode(x.unsqueeze(0))
        z = model.reparameterize(mu, logvar)
        latent_features.append(z.cpu().numpy().squeeze())

latent_features = np.array(latent_features)
np.save("results/easy/latent_features.npy", latent_features) 
print("Latent features saved:", latent_features.shape)
