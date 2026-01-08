import torch
import numpy as np
from torch import nn

output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
X_train = np.load(f"{output_folder}/X_train.npy")
y_train = np.load(f"{output_folder}/y_train.npy")

input_dim = X_train.shape[1]
latent_dim = 32
condition_dim = 1

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
model.load_state_dict(torch.load(f"{output_folder}/cvae_model_trained.pth"))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
c_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)

sample_x = X_train_tensor[:5].to(device)
sample_c = c_train_tensor[:5].to(device)

model.eval()
with torch.no_grad():
    recon_x, mu, logvar = model(sample_x, sample_c)

recon_x = recon_x.cpu().numpy()
original_x = sample_x.cpu().numpy()

for i in range(5):
    print(f"\nSample {i+1}:")
    print("Original (first 10 dims):", np.round(original_x[i][:10], 3))
    print("Reconstructed (first 10 dims):", np.round(recon_x[i][:10], 3))
