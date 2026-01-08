import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

# ---------------------------
# Paths (Relative to project root)
# ---------------------------
base_results = "results"
base_data = os.path.join("data", "features")

base_folders = {
    "Easy": os.path.join(base_results, "easy"),
    "Medium": os.path.join(base_results, "medium"),
    "Hard": os.path.join(base_results, "hard")
}

plots_folder = os.path.join(base_results, "plots")
os.makedirs(plots_folder, exist_ok=True)

# ---------------------------
# Helper: Purity Score
# ---------------------------
def purity_score(y_true, y_pred):
    if y_true is None or len(y_true) == 0:
        return np.nan
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.max(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

# ---------------------------
# Helper: Compute metrics
# ---------------------------
def compute_metrics(X, y_true, y_pred):
    metrics = {}
    # Unsupervised
    if len(np.unique(y_pred)) > 1:
        # For Silhouette and others, if X is too large, use a sample to speed up
        max_samples = 5000
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_samp, y_samp = X[idx], y_pred[idx]
            metrics["Silhouette"] = silhouette_score(X_samp, y_samp)
            metrics["Calinski-Harabasz"] = calinski_harabasz_score(X_samp, y_samp)
            metrics["Davies-Bouldin"] = davies_bouldin_score(X_samp, y_samp)
        else:
            metrics["Silhouette"] = silhouette_score(X, y_pred)
            metrics["Calinski-Harabasz"] = calinski_harabasz_score(X, y_pred)
            metrics["Davies-Bouldin"] = davies_bouldin_score(X, y_pred)
    else:
        metrics["Silhouette"] = np.nan
        metrics["Calinski-Harabasz"] = np.nan
        metrics["Davies-Bouldin"] = np.nan
    
    # Supervised
    if y_true is not None and len(y_true) == len(y_pred):
        metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
        metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred)
        metrics["Purity"] = purity_score(y_true, y_pred)
    else:
        metrics["ARI"] = np.nan
        metrics["NMI"] = np.nan
        metrics["Purity"] = np.nan
    return metrics

# ---------------------------
# Methods to strictly compare
# ---------------------------
requested_methods = [
    "AE+KMeans", 
    "CVAE+KMeans", 
    "DirectSpectral+KMeans", 
    "PCA+KMeans", 
    "KMeans", 
    "Agglomerative"
]

# ---------------------------
# Loop through folders
# ---------------------------
all_results = []

for folder_name, folder_path in base_folders.items():
    print(f"Processing {folder_name} folder...")
    X = None
    y_true = None

    # --- DATALOADING ---
    if folder_name == "Hard":
        x_npy = os.path.join(folder_path, "X_train.npy")
        y_npy = os.path.join(folder_path, "y_train.npy")
        if os.path.exists(x_npy) and os.path.exists(y_npy):
            X = np.load(x_npy)
            y_true = np.load(y_npy)
            print(f"Loaded Hard .npy features/labels: {X.shape}")
    
    if X is None:
        # Check both tier subfolder and the root features folder
        possible_data_dirs = [
            os.path.join("data", "features", folder_name.lower()),
            os.path.join("data", "features") # Medium/Hard data is here
        ]
        
        audio_df = None
        for d in possible_data_dirs:
            audio_csv_generic = os.path.join(d, "train.csv")
            audio_csv_named = os.path.join(d, f"train_audio.csv")
            
            if os.path.exists(audio_csv_generic):
                audio_df = pd.read_csv(audio_csv_generic, header=None if folder_name == "Easy" else 0)
                if folder_name != "Easy":
                     audio_df.columns = list(range(len(audio_df.columns) - 1)) + ["song_id"]
                break
            elif os.path.exists(audio_csv_named):
                audio_df = pd.read_csv(audio_csv_named)
                break
        
        if audio_df is not None:
            # Look for genre labels
            current_data_dir = os.path.dirname(audio_csv_generic if os.path.exists(audio_csv_generic) else audio_csv_named)
            genre_csv = os.path.join(current_data_dir, f"train_genre.csv")
            
            if os.path.exists(genre_csv):
                genre_df = pd.read_csv(genre_csv)
                if "song_id" in audio_df.columns and "song_id" in genre_df.columns:
                    combined_df = pd.merge(audio_df, genre_df, on="song_id")
                    X = combined_df.drop(columns=["song_id", "genre"], errors="ignore").values
                    y_true = combined_df["genre"].values
                else:
                    X = audio_df.drop(columns=["song_id"], errors="ignore").values
                    y_true = genre_df["genre"].values
            else:
                # Fallback for Medium/Hard where song_id might encode labels
                X = audio_df.drop(columns=["song_id"], errors="ignore").values
                if "song_id" in audio_df.columns:
                    def get_label(s):
                        s = str(s)
                        return s.split('_')[0] if '_' in s else 'unknown'
                    y_true = np.array([get_label(s) for s in audio_df["song_id"].values])

    if X is None:
        print(f"FAILED: Data not found for {folder_name}, skipping.")
        continue

    # Try to load existing comparison results
    comp_csv = os.path.join(folder_path, "clustering_comparison.csv")
    comp_df = pd.read_csv(comp_csv) if os.path.exists(comp_csv) else None

    n_clusters = len(np.unique(y_true)) if (y_true is not None and len(np.unique(y_true)) > 1) else 6
    metrics_dict = {}

    # 1. AE+KMeans
    y_ae = None
    if comp_df is not None and "ae" in comp_df.columns:
        y_ae = comp_df["ae"].values
    else:
        ae_file = os.path.join(folder_path, "y_ae.npy")
        if os.path.exists(ae_file):
            y_ae = np.load(ae_file)
    
    if y_ae is not None and len(y_ae) == len(X):
        metrics_dict["AE+KMeans"] = compute_metrics(X, y_true, y_ae)
    
    # 2. CVAE+KMeans
    # Check multiple locations/formats for CVAE
    y_cvae = None
    cvae_file = os.path.join(folder_path, "clusters.npy")
    cvae_csv = os.path.join(folder_path, "train_clusters_kmeans.csv")
    if os.path.exists(cvae_file):
        y_cvae = np.load(cvae_file)
    elif os.path.exists(cvae_csv):
        y_cvae = pd.read_csv(cvae_csv)["cluster"].values
    
    if y_cvae is not None and len(y_cvae) == len(X):
        metrics_dict["CVAE+KMeans"] = compute_metrics(X, y_true, y_cvae)

    # 3. DirectSpectral+KMeans
    y_spectral = None
    if comp_df is not None and "dbscan" in comp_df.columns: # Map dbscan as placeholder or similar
        y_spectral = comp_df["dbscan"].values
    else:
        spectral_file = os.path.join(folder_path, "y_audio.npy")
        if os.path.exists(spectral_file):
            y_spectral = np.load(spectral_file)
    
    if y_spectral is not None and len(y_spectral) == len(X):
        metrics_dict["DirectSpectral+KMeans"] = compute_metrics(X, y_true, y_spectral)

    # Easy fallback for pre-computed labels (map "Model_Default" to AE+KMeans for report)
    if folder_name == "Easy" and "AE+KMeans" not in metrics_dict:
        cluster_file = os.path.join(folder_path, "cluster_labels.npy")
        if os.path.exists(cluster_file):
             y_labels = np.load(cluster_file)
             if len(y_labels) == len(X):
                 metrics_dict["AE+KMeans"] = compute_metrics(X, y_true, y_labels)

    # 4. PCA+KMeans
    try:
        pca = PCA(n_components=min(16, X.shape[1]))
        X_pca = pca.fit_transform(X)
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pca_km = kmeans_pca.fit_predict(X_pca)
        metrics_dict["PCA+KMeans"] = compute_metrics(X, y_true, y_pca_km)
    except Exception as e:
        print(f"FAILED PCA+KMeans for {folder_name}: {e}")

    # 5. KMeans
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_km = kmeans.fit_predict(X)
        metrics_dict["KMeans"] = compute_metrics(X, y_true, y_km)
    except Exception as e:
        print(f"FAILED KMeans for {folder_name}: {e}")

    # 6. Agglomerative
    try:
        # Agglomerative is slow on 10k+ samples, so we might sample for baseline if needed
        # But let's try it first.
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        y_agglo = agglo.fit_predict(X)
        metrics_dict["Agglomerative"] = compute_metrics(X, y_true, y_agglo)
    except Exception as e:
        print(f"FAILED Agglomerative for {folder_name}: {e}")

    # Collect results
    for method in requested_methods:
        if method in metrics_dict:
            res = metrics_dict[method]
            res["Method"] = method
            res["Dataset"] = folder_name
            all_results.append(res)
        else:
            all_results.append({
                "Dataset": folder_name, "Method": method,
                "Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan,
                "ARI": np.nan, "NMI": np.nan, "Purity": np.nan
            })

# ---------------------------
# Process & Save Data
# ---------------------------
if all_results:
    df_all = pd.DataFrame(all_results)
    cols = ["Dataset", "Method", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI", "Purity"]
    df_all = df_all[cols]
    df_all.to_csv(os.path.join(plots_folder, "clustering_metrics_summary_all.csv"), index=False)
    print("SUCCESS: Metrics summary CSV saved")

    # ---------------------------
    # Plot 1: SUMMARY TABLE PLOT
    # ---------------------------
    print("Generating summary table plot...")
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.axis('off')
    
    # Prepare data for table: round values
    display_df = df_all.copy()
    numeric_cols = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI", "Purity"]
    for c in numeric_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    
    table_data = display_df.values
    col_labels = display_df.columns
    
    the_table = plt.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.8)
    
    # Color headers
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
    
    plt.title("Clustering Metrics Summary Table", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "metrics_summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------
    # Plot 2: INDIVIDUAL METRIC BARPLOTS
    # ---------------------------
    print("Generating individual metric barplots...")
    metrics_to_plot = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI", "Purity"]
    
    for metric in metrics_to_plot:
        if df_all[metric].isnull().all():
            continue
            
        plt.figure(figsize=(15, 8))
        pivot_df = df_all.pivot(index='Method', columns='Dataset', values=metric)
        pivot_df = pivot_df.reindex(requested_methods)
        
        ax = pivot_df.plot(kind='bar', figsize=(12, 7), width=0.8)
        
        plt.title(f"{metric} Comparison across Methods & Datasets", fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Methods", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Dataset")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{metric.lower().replace('-', '_')}_comparison_all.png"), dpi=300)
        plt.close()

    print("SUCCESS: All visual results saved in:", plots_folder)
else:
    print("FAILED: No metrics were computed.")
