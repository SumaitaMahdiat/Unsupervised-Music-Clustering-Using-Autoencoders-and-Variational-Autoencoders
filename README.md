# Unsupervised Music Clustering Using Autoencoders and Variational Autoencoders

This project explores deep representation learning for unsupervised music clustering. It compares traditional methods (PCA + K-Means) with deep generative models (AE, VAE, CVAE) across three tasks of increasing complexity, utilizing both audio features and lyrical embeddings.

## Project Overview
- **Objective**: Group music tracks based on similarity without utilizing explicit labels.
- **Complexity Levels**:
    - **Easy**: Audio-only feature learning using a Variational Autoencoder (VAE).
    - **Medium**: Multimodal learning by combining audio features and lyrics embeddings using an Autoencoder (AE).
    - **Hard**: Advanced feature learning using Conditional Variational Autoencoders (CVAE) and multiple clustering algorithms.

## Project Structure
```text
musicdata/
├── data/               # Preprocessed audio features, lyrics embeddings, and labels
├── src/                # Source code
│   ├── easy/           # VAE implementation and audio clustering
│   ├── medium/         # Multimodal AE implementation
│   └── hard/           # CVAE and advanced clustering techniques
├── results/            # Performance metrics and visualization plots
├── visuals.py          # Unified script for evaluation and plotting
├── report.tex          # Scientific report detailing methodology and results
└── README.md           # Project documentation
```

## Key Technologies
- **Deep Learning**: PyTorch (VAE, AE, CVAE)
- **Machine Learning**: Scikit-learn (K-Means, Spectral Clustering, Agglomerative Clustering, PCA)
- **Visualization**: Matplotlib, Seaborn, t-SNE, UMAP
- **Data Handling**: Pandas, NumPy


## Scripts Overview

### Root Directory
- `visuals.py`: The main entry point for unified evaluation. It processes results from all three tasks, computes standardized metrics, and generates comparison plots.
- `exploratory.ipynb`: Jupyter notebook for initial data analysis and feature distribution checks.

### Easy Task (`src/easy/`)
Focused on audio-only representation learning.
- `vae.py`: Implements the base Variational Autoencoder architecture.
- `trainer.py`: Handles the training loop for the audio VAE.
- `clustering.py`: Applies KMeans on the latent space and computes purity/silhouette scores.
- `separate_dataset.py`: Pre-processes the raw audio data and splits it into training/testing sets.
- `visual.py`: Generates the t-SNE and UMAP visualizations for Easy task clusters.

### Medium Task (`src/medium/`)
Introduces multimodal (audio + lyrics) data fusion.
- `convae.py`: Defines the multimodal Autoencoder/VAE architecture.
- `embed.py`: Utilizes `sentence-transformers` to convert raw song lyrics into numerical embeddings.
- `trainer.py`: Trains the model on concatenated audio and lyric features.
- `clustering.py`: Applies various clustering algorithms (KMeans, Agglomerative, DBSCAN) on the learned embeddings.
- `evaluate.py`: Standalone script for detailed quantitative evaluation of the medium task results.
- `visual1.py` & `visual2.py`: Provide detailed visualization of the latent space and clustering results.

### Hard Task (`src/hard/`)
Implements advanced conditioning and multiple clustering strategies.
- `cvae.py`: Implements the Conditional Variational Autoencoder (CVAE).
- `trainer.py`: Trains the CVAE using genre labels as conditional inputs.
- `clustering.py`: Experiments with Spectral and Agglomerative clustering on the CVAE embeddings.
- `dataset.py` & `final_dataset.py`: Handle complex data loading, genre mapping, and preprocessing for the hard task.
- `baseline.py`: Implements baseline models (like PCA or standard KMeans) for comparison.
- `evaluate.py`: Standalone evaluation script for hard task metrics.
- `reconstruct.py`: Visualizes the model's ability to reconstruct audio features from the latent space.
- `visual1.py`, `visual2.py`, & `visual3.py`: Comprehensive visualization scripts for analyzing genre-stratified clusters and latent space properties.

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training Models
You can train each level independently:
```bash
python src/easy/trainer.py
python src/medium/trainer.py
python src/hard/trainer.py
```

### 3. Evaluation & Visualization
To generate comparison plots and calculate quantitative metrics across all tasks:
```bash
python visuals.py
```

## Quantitative Results Highlights

| Task / Method | Silhouette | CH Index | ARI | NMI | Purity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Easy** (VAE + KMeans) | 0.102 | 380.7 | - | - | - |
| **Medium** (AE + KMeans) | 0.031 | 407.7 | -0.017 | 0.072 | 0.931 |
| **Hard** (CVAE + KMeans) | 0.044 | - | 0.032 | 0.059 | 0.557 |
| **Hard** (AE + KMeans) | 0.362 | - | 0.008 | 0.023 | 0.488 |

## Author
Sumaita Mahdiat
Department of CSE, BRAC University  
