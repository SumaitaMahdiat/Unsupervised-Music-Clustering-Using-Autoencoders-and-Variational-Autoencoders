# clustering.py
import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Paths
# ---------------------------
feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Load latent vectors
# ---------------------------
train_latent = pd.read_csv(os.path.join(feature_folder, "train_latent.csv"))
valid_latent = pd.read_csv(os.path.join(feature_folder, "valid_latent.csv"))
test_latent  = pd.read_csv(os.path.join(feature_folder, "test_latent.csv"))

# Keep song_id separate
train_ids = train_latent['song_id']
valid_ids = valid_latent['song_id']
test_ids  = test_latent['song_id']

X_train = train_latent.drop(columns=['song_id']).values
X_valid = valid_latent.drop(columns=['song_id']).values
X_test  = test_latent.drop(columns=['song_id']).values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------
# Clustering
# ---------------------------
def run_clustering(X, method='kmeans', n_clusters=5):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'agglo':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=1.5, min_samples=5)
    else:
        raise ValueError("Unsupported method")
    labels = model.fit_predict(X)
    return labels

# Example: Using 5 clusters (you can adjust)
n_clusters = 5

train_labels_km = run_clustering(X_train_scaled, 'kmeans', n_clusters)
valid_labels_km = run_clustering(X_valid_scaled, 'kmeans', n_clusters)
test_labels_km  = run_clustering(X_test_scaled, 'kmeans', n_clusters)

# ---------------------------
# Save cluster results
# ---------------------------
pd.DataFrame({'song_id': train_ids, 'cluster': train_labels_km}).to_csv(
    os.path.join(output_folder, 'train_clusters_kmeans.csv'), index=False
)
pd.DataFrame({'song_id': valid_ids, 'cluster': valid_labels_km}).to_csv(
    os.path.join(output_folder, 'valid_clusters_kmeans.csv'), index=False
)
pd.DataFrame({'song_id': test_ids, 'cluster': test_labels_km}).to_csv(
    os.path.join(output_folder, 'test_clusters_kmeans.csv'), index=False
)

print(" Clustering done and cluster labels saved.")
