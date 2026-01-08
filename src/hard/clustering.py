import torch
from sklearn.cluster import KMeans
import numpy as np
from torch import nn
import pickle

# Paths
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"

# Load data
X_train = np.load(f"{output_folder}/X_train.npy")
y_train = np.load(f"{output_folder}/y_train.npy")
with open(f"{output_folder}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

# Parameters
input_dim = X_train.shape[1]
latent_dim = 32
condition_dim = 1

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

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CVAE(input_dim, latent_dim, condition_dim).to(device)
model.load_state_dict(torch.load(f"{output_folder}/cvae_model_trained.pth"))
model.eval()

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
c_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)

with torch.no_grad():
    # Encode to latent space
    mu, logvar = model.encode(X_train_tensor, c_train_tensor)
    latent_space = mu.cpu().numpy()  # use mean as representation

print("Latent space shape:", latent_space.shape)

# ---------------------------
# K-Means clustering in latent space
# ---------------------------
num_clusters = len(le.classes_)  # number of genres
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(latent_space)

print("Cluster assignments for first 10 samples:", clusters[:10])

# Save latent space and clusters
np.save(f"{output_folder}/latent_space.npy", latent_space)
np.save(f"{output_folder}/clusters.npy", clusters)
