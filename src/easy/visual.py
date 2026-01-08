import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

latent_features = np.load("results/easy/latent_features.npy")
labels = np.load("results/easy/cluster_labels.npy")

tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(latent_features)

plt.figure(figsize=(8,6))
scatter = plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=labels, cmap='tab10', s=50)
plt.title("t-SNE of VAE Latent Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label='Cluster')
plt.savefig("results/easy/tsne_clusters.png")  
plt.show()
print("t-SNE plot saved to results/easy/tsne_clusters.png")
umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(latent_features)

plt.figure(figsize=(8,6))
scatter = plt.scatter(umap_proj[:,0], umap_proj[:,1], c=labels, cmap='tab10', s=50)
plt.title("UMAP of VAE Latent Features")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label='Cluster')
plt.savefig("results/easy/umap_clusters.png") 
plt.show()
print("UMAP plot saved to results/easy/umap_clusters.png")
