import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE, vae_loss

# ---------------------------
# Paths to CSVs
# ---------------------------
train_csv = "C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/train.csv"
valid_csv = "C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/valid.csv"

# ---------------------------
# Datasets and DataLoaders
# ---------------------------
train_dataset = MusicDataset(train_csv)
valid_dataset = MusicDataset(valid_csv)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# ---------------------------
# Device and model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim=train_dataset.features.shape[1], latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_dataset)
    
    # Optional: validation loss
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x in valid_loader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_dataset)
    
    print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_loss:.4f}  Valid Loss: {avg_valid_loss:.4f}")

# ---------------------------
# Save model to results/easy
# ---------------------------
torch.save(model.state_dict(), "results/easy/vae_music.pth")
print("Model saved as vae_music.pth")
