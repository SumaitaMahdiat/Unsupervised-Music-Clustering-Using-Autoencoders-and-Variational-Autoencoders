import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
os.makedirs(output_folder, exist_ok=True)

train_latent = pd.read_csv(os.path.join(feature_folder, "train_latent.csv"))
train_clusters = pd.read_csv(os.path.join(output_folder, "train_clusters_kmeans.csv"))

X = train_latent.drop(columns=['song_id']).values
labels = train_clusters['cluster'].values

def reduce_dim(X, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError("Unsupported method")
    return X_2d

X_2d = reduce_dim(X, method='pca')  
plt.figure(figsize=(8,6))
for cluster in set(labels):
    idx = labels == cluster
    plt.scatter(X_2d[idx,0], X_2d[idx,1], label=f'Cluster {cluster}', alpha=0.7)
plt.title("Latent Space Clustering Visualization")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "latent_clusters.png"))
plt.show()
print(" Latent space plot saved.")

X_2d_tsne = reduce_dim(X, method='tsne')
plt.figure(figsize=(8,6))
for cluster in set(labels):
    idx = labels == cluster
    plt.scatter(X_2d_tsne[idx,0], X_2d_tsne[idx,1], label=f'Cluster {cluster}', alpha=0.7)
plt.title("Latent Space Clustering Visualization (t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "latent_clusters_tsne.png"))
plt.show()
print("t-SNE latent space plot saved.")

X_2d_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(8,6))
for cluster in set(labels):
    idx = labels == cluster
    plt.scatter(X_2d_umap[idx,0], X_2d_umap[idx,1], label=f'Cluster {cluster}', alpha=0.7)
plt.title("Latent Space Clustering Visualization (UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "latent_clusters_umap.png"))
plt.show()
print(" UMAP latent space plot saved.")
