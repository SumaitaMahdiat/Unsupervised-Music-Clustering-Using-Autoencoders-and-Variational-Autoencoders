import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
os.makedirs(feature_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

class MusicDataset(Dataset):
    def __init__(self, audio_csv, lyrics_emb_csv):
        self.audio_df = pd.read_csv(audio_csv)
        self.lyrics_df = pd.read_csv(lyrics_emb_csv)
        audio_id_col = self.audio_df.columns[-1]
        lyrics_id_col = self.lyrics_df.columns[-1]
        self.audio_df = self.audio_df.rename(columns={audio_id_col: "song_id"})
        self.lyrics_df = self.lyrics_df.rename(columns={lyrics_id_col: "song_id"})
        self.data = self.audio_df.merge(self.lyrics_df, on="song_id")

        self.audio_feats = (
            self.data.select_dtypes(include=[np.number])
            .iloc[:, : self.audio_df.shape[1] - 1]
            .values.astype(np.float32)
        )

        self.lyrics_embs = (
            self.data.select_dtypes(include=[np.number])
            .iloc[:, self.audio_df.shape[1] - 1 :]
            .values.astype(np.float32)
        )

        self.song_ids = self.data["song_id"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.audio_feats[idx]
        lyrics = self.lyrics_embs[idx]
        song_id = self.data.iloc[idx]['song_id']
        return torch.tensor(audio), torch.tensor(lyrics), song_id

class ConvVAE(nn.Module):
    def __init__(self, audio_dim, lyrics_dim, latent_dim=64):
        super(ConvVAE, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, audio_dim + lyrics_dim)
        )

    def encode(self, x_audio, x_lyrics):
        h_audio = self.audio_encoder(x_audio)
        h_lyrics = self.lyrics_encoder(x_lyrics)
        h = torch.cat([h_audio, h_lyrics], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_audio, x_lyrics):
        mu, logvar = self.encode(x_audio, x_lyrics)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld

def train_vae(model, dataloader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for audio, lyrics, _ in dataloader:
            audio, lyrics = audio.float(), lyrics.float()
            x = torch.cat([audio, lyrics], dim=1)
            optimizer.zero_grad()
            recon, mu, logvar = model(audio, lyrics)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train_dataset = MusicDataset(
        os.path.join(feature_folder, "train_audio.csv"),
        os.path.join(feature_folder, "train_lyrics_emb.csv")
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    audio_dim = train_dataset.audio_feats.shape[1]
    lyrics_dim = train_dataset.lyrics_embs.shape[1]
    model = ConvVAE(audio_dim, lyrics_dim, latent_dim=64)
    train_vae(model, train_loader, epochs=50, lr=1e-3)
  
    torch.save(model.state_dict(), os.path.join(output_folder, "cvae.pth"))
    print("CVAE trained and saved.")
