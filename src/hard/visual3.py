import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

output_folder = r"C:/Users/user/OneDrive/Documents/musicdata/results/hard"
plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

y_train      = np.load(f"{output_folder}/y_train.npy")     
clusters_cvae = np.load(f"{output_folder}/clusters.npy")     
y_ae         = np.load(f"{output_folder}/y_ae.npy")         
y_pca        = np.load(f"{output_folder}/y_pca.npy")        
y_audio      = np.load(f"{output_folder}/y_audio.npy")      
genres = np.unique(y_train)

def save_cluster_distribution(cluster_labels, true_genres, method_name, filename):
    df = pd.DataFrame({"Cluster": cluster_labels, "Genre": true_genres})
    crosstab = pd.crosstab(df["Cluster"], df["Genre"])
    crosstab.plot(kind="bar", stacked=True, figsize=(8,6), colormap="tab20")
    plt.title(f"{method_name} - Cluster Distribution Over Genres")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Samples")
    plt.legend(title="Genre", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, filename), dpi=300)
    plt.close()

save_cluster_distribution(y_ae, y_train, "AE + KMeans", "ae_cluster_distribution.png")
save_cluster_distribution(y_audio, y_train, "Audio Features + KMeans", "audio_cluster_distribution.png")
save_cluster_distribution(clusters_cvae, y_train, "CVAE + KMeans", "cvae_cluster_distribution.png")
save_cluster_distribution(y_pca, y_train, "PCA + KMeans", "pca_cluster_distribution.png")
print(" All cluster distribution plots saved in:", plots_folder)
