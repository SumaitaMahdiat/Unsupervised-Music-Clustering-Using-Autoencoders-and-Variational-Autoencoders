# Unsupervised Music Clustering Using Autoencoders and Variational Autoencoders
This project explores deep representation learning for unsupervised music clustering. It compares traditional methods (PCA + K-Means) with deep generative models (AE, VAE, CVAE) across three tasks of increasing complexity, utilizing both audio features and lyrical embeddings.

## Project Overview
- Objective: Group music tracks based on similarity without utilizing explicit labels.
- Complexity Levels:
    - Easy: Audio-only feature learning using a Variational Autoencoder (VAE).
    - Medium: Multimodal learning by combining audio features and lyrics embeddings using an Autoencoder (AE).
    - Hard: Advanced feature learning using Conditional Variational Autoencoders (CVAE) and multiple clustering algorithms.

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
- Deep Learning: PyTorch (VAE, AE, CVAE)
- Machine Learning: Scikit-learn (K-Means, Spectral Clustering, Agglomerative Clustering, PCA)
- Visualization: Matplotlib, Seaborn, t-SNE, UMAP
- Data Handling: Pandas, NumPy

## Installation
Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

### Training Models
Each task has its own training script:
```bash
python src/easy/trainer.py
python src/medium/trainer.py
python src/hard/trainer.py
```

### Evaluation & Visualization
To generate comparison plots and calculate quantitative metrics:
```bash
python visuals.py
```

## Quantitative Results Highlights
| Task / Method | Silhouette | CH Index | ARI | NMI | Purity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Easy (VAE + KMeans) | 0.102 | 380.7 | - | - | - |
| Medium (AE + KMeans) | 0.031 | 407.7 | -0.017 | 0.072 | 0.931 |
| Hard (CVAE + KMeans) | 0.044 | - | 0.032 | 0.059 | 0.557 |
| Hard (AE + KMeans) | 0.362 | - | 0.008 | 0.023 | 0.488 |

## Author
Sumaita Mahdiat
Department of CSE, BRAC University  
