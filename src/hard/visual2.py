import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os

output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

latent_cvae = np.load(f"{output_folder}/latent_space.npy")
latent_ae   = np.load(f"{output_folder}/latent_ae.npy")
X_pca       = np.load(f"{output_folder}/X_pca.npy")
audio_train = pd.read_pickle(f"{output_folder}/audio_train.pkl")

clusters_cvae = np.load(f"{output_folder}/clusters.npy")
y_ae         = np.load(f"{output_folder}/y_ae.npy")
y_pca        = np.load(f"{output_folder}/y_pca.npy")
y_audio      = np.load(f"{output_folder}/y_audio.npy")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)

latent_cvae_2d = tsne.fit_transform(latent_cvae)
latent_ae_2d   = tsne.fit_transform(latent_ae)
X_pca_2d       = tsne.fit_transform(X_pca)
audio_2d       = tsne.fit_transform(audio_train.values)

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

save_tsne_plot(latent_cvae_2d, clusters_cvae, "CVAE Latent Space Clustering", "cvae_latent_clusters.png")
save_tsne_plot(latent_ae_2d, y_ae, "Autoencoder + K-Means Clustering", "ae_latent_clusters.png")
save_tsne_plot(X_pca_2d, y_pca, "PCA + K-Means Clustering", "pca_clusters.png")
save_tsne_plot(audio_2d, y_audio, "Audio Features + K-Means Clustering", "audio_clusters.png")

print(" All 4 t-SNE comparison plots saved in:", plots_folder)

import umap
umap_reducer = umap.UMAP(n_components=2, random_state=42)
latent_cvae_umap = umap_reducer.fit_transform(latent_cvae)
save_tsne_plot(latent_cvae_umap, clusters_cvae, "CVAE Latent Space Clustering (UMAP)", "cvae_latent_clusters_umap.png")
print("UMAP plot saved in:", plots_folder)