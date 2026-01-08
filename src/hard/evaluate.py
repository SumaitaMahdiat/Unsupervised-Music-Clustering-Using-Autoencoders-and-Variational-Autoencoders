from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
import numpy as np

# Paths
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"

# Load data
y_train = np.load(f"{output_folder}/y_train.npy")
latent_space = np.load(f"{output_folder}/latent_space.npy")
clusters = np.load(f"{output_folder}/clusters.npy")

# True labels
y_true = y_train  # genre labels

# Predicted clusters
y_pred = clusters

# ---------------------------
# Silhouette Score
# ---------------------------
sil_score = silhouette_score(latent_space, y_pred)
print(f"Silhouette Score: {sil_score:.3f}")

# ---------------------------
# Normalized Mutual Information (NMI)
# ---------------------------
nmi_score = normalized_mutual_info_score(y_true, y_pred)
print(f"NMI: {nmi_score:.3f}")

# ---------------------------
# Adjusted Rand Index (ARI)
# ---------------------------
ari_score = adjusted_rand_score(y_true, y_pred)
print(f"ARI: {ari_score:.3f}")

# ---------------------------
# Cluster Purity
# ---------------------------
def cluster_purity(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    purity = df.groupby('y_pred')['y_true'].apply(lambda x: x.value_counts().max()/len(x)).mean()
    return purity

purity = cluster_purity(y_true, y_pred)
print(f"Cluster Purity: {purity:.3f}")
