from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import torch
import random
import os

output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
X_train = np.load(f"{output_folder}/X_train.npy")
y_train = np.load(f"{output_folder}/y_train.npy")
audio_train = pd.read_pickle(f"{output_folder}/audio_train.pkl")
with open(f"{output_folder}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

input_dim = X_train.shape[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

def cluster_purity(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    return df.groupby('y_pred')['y_true'].apply(lambda x: x.value_counts().max()/len(x)).mean()

num_clusters = len(le.classes_)

pca = PCA(n_components=32)
X_pca = pca.fit_transform(X_train)
kmeans_pca = KMeans(n_clusters=num_clusters, random_state=42)
y_pca = kmeans_pca.fit_predict(X_pca)

print("\n--- PCA + K-Means ---")
sample_size = min(1000, len(X_pca))
sample_indices = random.sample(range(len(X_pca)), sample_size)
print("Silhouette:", round(silhouette_score(X_pca[sample_indices], y_pca[sample_indices]),3))
print("NMI:", round(normalized_mutual_info_score(y_train, y_pca),3))
print("ARI:", round(adjusted_rand_score(y_train, y_pca),3))
print("Purity:", round(cluster_purity(y_train, y_pca),3))

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

ae_model = AE(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
epochs = 50

for epoch in range(epochs):
    ae_model.train()
    optimizer.zero_grad()
    recon, _ = ae_model(X_tensor)
    loss = nn.MSELoss()(recon, X_tensor)
    loss.backward()
    optimizer.step()

ae_model.eval()
with torch.no_grad():
    _, latent_ae = ae_model(X_tensor)
latent_ae = latent_ae.cpu().numpy()

kmeans_ae = KMeans(n_clusters=num_clusters, random_state=42)
y_ae = kmeans_ae.fit_predict(latent_ae)

print("\n--- Autoencoder + K-Means ---")
sample_indices = random.sample(range(len(latent_ae)), sample_size)
print("Silhouette:", round(silhouette_score(latent_ae[sample_indices], y_ae[sample_indices]),3))
print("NMI:", round(normalized_mutual_info_score(y_train, y_ae),3))
print("ARI:", round(adjusted_rand_score(y_train, y_ae),3))
print("Purity:", round(cluster_purity(y_train, y_ae),3))

kmeans_audio = KMeans(n_clusters=num_clusters, random_state=42)
y_audio = kmeans_audio.fit_predict(audio_train.values)

print("\n--- Direct Audio Feature K-Means ---")
sample_indices = random.sample(range(len(audio_train)), sample_size)
print("Silhouette:", round(silhouette_score(audio_train.values[sample_indices], y_audio[sample_indices]),3))
print("NMI:", round(normalized_mutual_info_score(y_train, y_audio),3))
print("ARI:", round(adjusted_rand_score(y_train, y_audio),3))
print("Purity:", round(cluster_purity(y_train, y_audio),3))

np.save(f"{output_folder}/latent_ae.npy", latent_ae)
np.save(f"{output_folder}/y_ae.npy", y_ae)
np.save(f"{output_folder}/X_pca.npy", X_pca)
np.save(f"{output_folder}/y_pca.npy", y_pca)
np.save(f"{output_folder}/y_audio.npy", y_audio)