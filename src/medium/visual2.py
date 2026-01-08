import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
output_folder = feature_folder
os.makedirs(output_folder, exist_ok=True)

try:
    train_latent = pd.read_csv(os.path.join(feature_folder, "train_latent.csv"))
except FileNotFoundError:
    print(f"Error: train_latent.csv not found in {feature_folder}. Run trainer.py first.")
    import sys
    sys.exit()

def get_label(s):
    s = str(s)
    return s.split('_')[0] if '_' in s else 'unknown'

labels = np.array([get_label(s) for s in train_latent['song_id'].values])
unique_gt_labels = np.unique(labels)
n_clusters_gt = len(unique_gt_labels)
X = train_latent.drop(columns=['song_id']).values

MAX_SAMPLES = 5000
if len(X) > MAX_SAMPLES:
    print(f"Subsampling from {len(X)} to {MAX_SAMPLES} for performance...")
    indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X = X[indices]
    labels = labels[indices]
    subsampled_ids = train_latent['song_id'].values[indices]
else:
    subsampled_ids = train_latent['song_id'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
def plot_clusters(X_2d, cluster_labels, title, save_path):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(cluster_labels)
   
    non_noise_labels = [l for l in unique_labels if str(l) != '-1']
    n_colors = max(len(non_noise_labels), 1)
    
    palette = sns.color_palette("husl", n_colors)
    color_map = {str(label): palette[i] for i, label in enumerate(non_noise_labels)}
    
    for label in unique_labels:
        mask = (cluster_labels == label)
        points = X_2d[mask]
        
        label_str = str(label)
        if label_str == '-1':
            color = "gray"
            label_name = "Noise (DBSCAN)"
        else:
            color = color_map.get(label_str, "black")
            label_name = f"Cluster {label_str}" if isinstance(label, (int, np.integer)) else label_str
            
        plt.scatter(points[:, 0], points[:, 1], color=color, alpha=0.6, label=label_name, s=30)

    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    
    if len(unique_labels) > 15:
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    else:
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("Plotting Ground Truth...")
plot_clusters(X_2d, labels, "Ground Truth (Language Folders)",
              os.path.join(output_folder, "ground_truth.png"))

print(f"Running KMeans with n_clusters={n_clusters_gt}...")
kmeans = KMeans(n_clusters=n_clusters_gt, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
plot_clusters(X_2d, kmeans_labels, f"KMeans Clusters (k={n_clusters_gt})",
              os.path.join(output_folder, "kmeans_clusters.png"))

print("Running Agglomerative Clustering...")
agglo = AgglomerativeClustering(n_clusters=n_clusters_gt)
agglo_labels = agglo.fit_predict(X_scaled)
plot_clusters(X_2d, agglo_labels, f"Agglomerative Clusters (k={n_clusters_gt})",
              os.path.join(output_folder, "agglo_clusters.png"))

print("Running DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
plot_clusters(X_2d, dbscan_labels, "DBSCAN Clusters",
              os.path.join(output_folder, "dbscan_clusters.png"))

pd.DataFrame({
    'song_id': subsampled_ids,
    'ground_truth': labels,
    'kmeans': kmeans_labels,
    'agglo': agglo_labels,
    'dbscan': dbscan_labels
}).to_csv(os.path.join(output_folder, "clustering_comparison.csv"), index=False)
print(f"All plots and comparison CSV saved to {output_folder}")
