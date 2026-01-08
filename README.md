# Unsupervised Music Clustering Using Autoencoders and Variational Autoencoders

[![Project Status](https://img.shields.io/badge/Status-Complete-green.svg)](https://github.com/yourusername/musicdata)
[![LaTeX](https://img.shields.io/badge/LaTeX-Report-blue.svg)](report.tex)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](requirements.txt)

This project explores **deep representation learning** for unsupervised music clustering. It compares traditional methods (PCA + K-Means) with deep generative models (AE, VAE, CVAE) across three tasks of increasing complexity, utilizing both audio features and lyrical embeddings.

---

## Project Overview

The system learns compact latent representations from music data, applies multiple clustering algorithms, evaluates cluster quality using standard metrics, and visualizes latent spaces and reconstruction behavior.

### Complexity Levels

*   **Easy Task**: Audio-only feature learning using a **Variational Autoencoder (VAE)**.
*   **Medium Task**: Multimodal learning combining audio features and lyric embeddings using a **Multimodal VAE (ConvVAE)**.
*   **Hard Task**: Advanced feature learning using **Conditional Variational Autoencoders (CVAE)** with genre conditioning.

---

## Repository/Folder Structure

```text
musicdata/
├── data/               # Raw and preprocessed audio/lyrics data
├── src/                # Source code
│   ├── easy/           # Audio-only VAE pipeline
│   │   ├── separate_dataset.py  # PREPARATION: Audio preprocessing & split
│   │   ├── vae.py               # VAE Architecture
│   │   ├── trainer.py           # VAE Training
│   │   └── ...
│   ├── medium/         # Multimodal VAE pipeline
│   │   ├── embed.py             # PREPARATION: Lyrics embedding
│   │   ├── convae.py            # Multimodal VAE Architecture
│   │   └── ...
│   └── hard/           # CVAE and advanced clustering
│       ├── final_dataset.py     # PREPARATION: Unified dataset processing
│       ├── cvae.py              # CVAE Architecture
│       └── ...
├── results/            # Performance metrics and visualization plots
├── visuals.py          # Unified evaluation & plotting script
├── report.tex          # Scientific report (LaTeX)
└── README.md           # Project documentation
```

---
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

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/musicdata.git
    cd musicdata
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Instructions

### 1. Data Preparation (CRITICAL)

Before training the models, you must prepare the datasets for each task level.

#### **Easy Task**
To extract MFCC features and split the audio data for the Easy task:
```bash
python src/easy/separate_dataset.py
```

#### **Medium & Hard Tasks**
To process the unified dataset (audio + lyrics + genres) and generate embeddings:
1.  **Extract Features**:
    ```bash
    python src/hard/final_dataset.py
    ```
2.  **Generate Lyrics Embeddings**:
    ```bash
    python src/medium/embed.py
    ```

### 2. Training Models
Each task can be trained independently once the data is prepared:
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

---

## Quantitative Results Highlights

The following table summarizes the performance of various methods across the three tasks:

| Task / Method | Silhouette | CH Index | ARI | NMI | Purity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Easy** (PCA + KMeans) | 0.196 | 1901.8 | - | - | - |
| **Medium** (PCA + KMeans) | 0.179 | 1331.0 | 0.005 | 0.022 | 0.473 |
| **Hard** (CVAE + KMeans) | -0.018 | 1051.1 | 0.032 | 0.059 | 0.558 |
| **Hard** (AE + KMeans) | 0.144 | 2755.3 | 0.008 | 0.023 | 0.490 |
| **Hard** (PCA + KMeans) | 0.184 | 3131.2 | 0.006 | 0.022 | 0.476 |

---

## Visualizations
The project generates several types of visualizations saved in the `results/` directory:
- **Latent Space**: t-SNE and UMAP projections of learned embeddings.
- **Metric Comparison**: Bar plots for Silhouette, ARI, NMI, and Purity across tasks.
- **Reconstruction**: Original vs. Reconstructed audio features (Hard task).

---

## Author
**Sumaita Mahdiat**  
Department of Computer Science and Engineering  
BRAC University
