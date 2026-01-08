# trainer.py
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from convae import ConvVAE, MusicDataset
import numpy as np

feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/data/features"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
model_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
model_path = os.path.join(model_folder, "cvae.pth")
os.makedirs(output_folder, exist_ok=True)

def extract_latents(model, dataloader, device='cpu'):
    model.eval()
    latents = []
    song_ids = []

    with torch.no_grad():
        for audio, lyrics, ids in dataloader:
            audio = audio.float().to(device)
            lyrics = lyrics.float().to(device)

            mu, logvar = model.encode(audio, lyrics)
            z = mu

            latents.append(z.cpu().numpy())
            song_ids.extend(ids)

    latents = np.vstack(latents)

    return pd.DataFrame(
        latents,
        columns=[f'z{i}' for i in range(latents.shape[1])]
    ).assign(song_id=song_ids)

train_dataset = MusicDataset(
    os.path.join(feature_folder, "train_audio.csv"),
    os.path.join(feature_folder, "train_lyrics_emb.csv")
)
valid_dataset = MusicDataset(
    os.path.join(feature_folder, "valid_audio.csv"),
    os.path.join(feature_folder, "valid_lyrics_emb.csv")
)
test_dataset = MusicDataset(
    os.path.join(feature_folder, "test_audio.csv"),
    os.path.join(feature_folder, "test_lyrics_emb.csv")
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

audio_dim = train_dataset.audio_feats.shape[1]
lyrics_dim = train_dataset.lyrics_embs.shape[1]
model = ConvVAE(audio_dim, lyrics_dim, latent_dim=64)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

train_latent = extract_latents(model, train_loader)
valid_latent = extract_latents(model, valid_loader)
test_latent  = extract_latents(model, test_loader)

train_latent.to_csv(os.path.join(output_folder, "train_latent.csv"), index=False)
valid_latent.to_csv(os.path.join(output_folder, "valid_latent.csv"), index=False)
test_latent.to_csv(os.path.join(output_folder, "test_latent.csv"), index=False)
print(" Latent vectors saved for clustering.")
