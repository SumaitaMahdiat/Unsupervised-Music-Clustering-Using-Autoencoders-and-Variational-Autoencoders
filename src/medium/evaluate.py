# evaluate.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.metrics.cluster import contingency_matrix

# ---------------------------
# Paths
# ---------------------------
feature_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"
output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/medium"

# ---------------------------
# Helper: Purity Score
# ---------------------------
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

# ---------------------------
# Load latent vectors & cluster labels
# ---------------------------
def load_data(split):
    latent_df = pd.read_csv(os.path.join(feature_folder, f"{split}_latent.csv"))
    cluster_df = pd.read_csv(os.path.join(output_folder, f"{split}_clusters_kmeans.csv"))
    
    # Merge on song_id just in case ordering changed
    merged = latent_df.merge(cluster_df, on='song_id')
    
    X = merged.drop(columns=['song_id', 'cluster']).values
    labels = merged['cluster'].values
    ids = merged['song_id'].values
    
    return X, labels, ids

# ---------------------------
# Ground Truth extraction
# ---------------------------
def get_ground_truth(ids):
    # Prefix-based ground truth (e.g., 'german_song1' -> 'german')
    return np.array([str(s).split('_')[0] if '_' in str(s) else 'unknown' for s in ids])

# ---------------------------
# Evaluation metrics
# ---------------------------
def evaluate_all(X, labels, gt_labels, max_samples=5000):
    # Remove noise labels if any
    mask = labels != -1
    X, labels, gt_labels = X[mask], labels[mask], gt_labels[mask]

    # Subsample for expensive metrics if needed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sub, labels_sub = X[idx], labels[idx]
    else:
        X_sub, labels_sub = X, labels

    metrics = {}
    metrics['Silhouette'] = silhouette_score(X_sub, labels_sub)
    metrics['CH Score'] = calinski_harabasz_score(X_sub, labels_sub)
    metrics['DB Index'] = davies_bouldin_score(X_sub, labels_sub)
    metrics['ARI'] = adjusted_rand_score(gt_labels, labels)
    metrics['NMI'] = normalized_mutual_info_score(gt_labels, labels)
    metrics['Purity'] = purity_score(gt_labels, labels)
    
    return metrics

# ---------------------------
# Main Evaluation
# ---------------------------
for split in ['train', 'valid', 'test']:
    try:
        X, labels, ids = load_data(split)
        gt_labels = get_ground_truth(ids)
        
        print(f"\n=== {split.upper()} Clustering Evaluation ===")
        metrics = evaluate_all(X, labels, gt_labels)
        
        for name, val in metrics.items():
            print(f"{name:15}: {val:.4f}")
            
    except FileNotFoundError as e:
        print(f"Skipping {split}: {e}")

print("\nEvaluation complete.")
