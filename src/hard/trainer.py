import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from torch import nn

output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
X_train = np.load(f"{output_folder}/X_train.npy")
y_train = np.load(f"{output_folder}/y_train.npy")
X_valid = np.load(f"{output_folder}/X_valid.npy")
y_valid = np.load(f"{output_folder}/y_valid.npy")
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
model.load_state_dict(torch.load(f"{output_folder}/cvae_model.pth"))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
c_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32) 

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
c_valid_tensor = torch.tensor(y_valid.reshape(-1,1), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, c_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
beta = 1.0  

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, cb in train_loader:
        xb, cb = xb.to(device), cb.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(xb, cb)
        loss = loss_function(recon, xb, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print(" Training complete")
torch.save(model.state_dict(), f"{output_folder}/cvae_model_trained.pth")
