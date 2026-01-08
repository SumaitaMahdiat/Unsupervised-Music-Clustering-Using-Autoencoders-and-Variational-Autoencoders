import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# ---------------------------
# Load latent features
# ---------------------------
latent_features = np.load("results/easy/latent_features.npy")

# ---------------------------
# K-Means clustering
# ---------------------------
n_clusters = 6  # adjust based on your dataset

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(latent_features)

sil = silhouette_score(latent_features, labels)
ch = calinski_harabasz_score(latent_features, labels)
print(f"K-Means on VAE Latent Features: Silhouette={sil:.4f}, CH={ch:.4f}")

# ---------------------------
# PCA + K-Means baseline
# ---------------------------
pca_features = PCA(n_components=16).fit_transform(latent_features)
labels_pca = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pca_features)
sil_pca = silhouette_score(pca_features, labels_pca)
ch_pca = calinski_harabasz_score(pca_features, labels_pca)
print(f"PCA + K-Means: Silhouette={sil_pca:.4f}, CH={ch_pca:.4f}")

# ---------------------------
# Save cluster labels
# ---------------------------
np.save("results/easy/cluster_labels.npy", labels)  # Save K-Means cluster labels to results/easy
 