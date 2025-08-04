

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy import sparse


#  Setup 

os.makedirs("visualizations", exist_ok=True)

logging.basicConfig(
    filename="logs/visualization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# TF-IDF Visualization 
def visualize_tfidf_pca():
    """
    Loads TF-IDF vectors, normalizes them, reduces dimensionality via PCA,
    and plots a 2D scatter plot.
    """
    try:
        logging.info("Loading TF-IDF matrix")
        tfidf = sparse.load_npz("features/tfidf_vectors.npz")
        tfidf_norm = normalize(tfidf, norm='l2')

        logging.info("🧪 Running PCA on TF-IDF")
        reduced = PCA(n_components=2).fit_transform(tfidf_norm[:3000].toarray())

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.4)
        plt.title("TF-IDF Tweet Vectors (PCA projection)")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("visualizations/tfidf_pca_plot.png")
        logging.info("Saved TF-IDF plot to visualizations/tfidf_pca_plot.png")
        

    except Exception as e:
        logging.error(f"TF-IDF visualization failed: {e}")
        print("TF-IDF visualization error:", e)

# Embedding Visualization 
def visualize_embeddings_pca():
    """
    Loads sentence embeddings from file, applies PCA to reduce to 2D,
    and plots a scatter plot.
    """
    try:
        logging.info("Loading sentence embeddings")
        df = pd.read_parquet("embeddings/tweets_with_embeddings.parquet")
        emb = np.array(df["embedding"].to_list())

        logging.info("Running PCA on embeddings")
        reduced = PCA(n_components=2).fit_transform(emb)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.4)
        plt.title("Sentence Embeddings (PCA projection)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("visualizations/embedding_pca_plot.png")
        logging.info("Saved embeddings plot to visualizations/embedding_pca_plot.png")
        

    except Exception as e:
        logging.error(f"Embedding visualization failed: {e}")
        print("Embedding visualization error:", e)


# Main 
def visualize():
    """
    Generate PCA-based visualizations for:
    - TF-IDF sparse vectors
    - Sentence transformer embeddings
    """
    logging.info("Starting visualization pipeline")
    visualize_tfidf_pca()
    visualize_embeddings_pca()
    logging.info(" Visualization complete")

if __name__ == "__main__":
    visualize()
