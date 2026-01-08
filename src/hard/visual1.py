import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import torch
from torch import nn

# ---------------------------
# Paths
# ---------------------------
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
latent_space = np.load(f"{output_folder}/latent_space.npy")
clusters     = np.load(f"{output_folder}/clusters.npy")
y_train      = np.load(f"{output_folder}/y_train.npy")
audio_train  = pd.read_pickle(f"{output_folder}/audio_train.pkl")
X_train      = np.load(f"{output_folder}/X_train.npy")

# ---------------------------
# CVAE model setup
# ---------------------------
input_dim     = X_train.shape[1]
latent_dim    = 32
condition_dim = 1

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 256)
        self.fc2_mu = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
        self.relu = nn.ReLU()
    
    def encode(self, x, c):
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
model.load_state_dict(torch.load(f"{output_folder}/cvae_model_trained.pth", map_location=device))

# ---------------------------
# Forward pass for first 5 samples
# ---------------------------
X_train_tensor = torch.tensor(X_train[:5], dtype=torch.float32).to(device)
c_train_tensor = torch.tensor(y_train[:5].reshape(-1,1), dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    recon_x, _, _ = model(X_train_tensor, c_train_tensor)
recon_x     = recon_x.cpu().numpy()
original_x  = X_train_tensor.cpu().numpy()

# ---------------------------
# t-SNE latent 2D
# ---------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(latent_space)

# Helper to save t-SNE plots
def save_tsne_plot(latent_2d, colors, title, filename):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latent_2d[:,0], latent_2d[:,1], c=colors, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, filename), dpi=300)
    plt.close()

# Save t-SNE plots
save_tsne_plot(latent_2d, clusters, "CVAE Latent Space Clustering", "cvae_latent_clusters.png")
save_tsne_plot(latent_2d, y_train, "CVAE Latent Space Colored by Genre", "cvae_latent_genres.png")

# ---------------------------
# Helper to save reconstruction plots
# ---------------------------
def save_reconstruction_plot(original, reconstructed, sample_idx, feature_type):
    plt.figure(figsize=(8,4))
    plt.plot(original, label=f"Original {feature_type}")
    plt.plot(reconstructed, label=f"Reconstructed {feature_type}")
    plt.title(f"{feature_type} Reconstruction - Sample {sample_idx+1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{feature_type.lower()}_reconstruction_sample{sample_idx+1}.png"), dpi=300)
    plt.close()

# First sample reconstruction
save_reconstruction_plot(original_x[0][:audio_train.shape[1]], 
                         recon_x[0][:audio_train.shape[1]], 0, "Audio")
save_reconstruction_plot(original_x[0][audio_train.shape[1]:], 
                         recon_x[0][audio_train.shape[1]:], 0, "Latent")

# All 5 samples
for i in range(5):
    save_reconstruction_plot(original_x[i][:audio_train.shape[1]], 
                             recon_x[i][:audio_train.shape[1]], i, "Audio")
    save_reconstruction_plot(original_x[i][audio_train.shape[1]:], 
                             recon_x[i][audio_train.shape[1]:], i, "Latent")

print(" All CVAE latent and reconstruction plots saved in:", plots_folder)
