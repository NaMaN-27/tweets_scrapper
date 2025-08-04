
import pandas as pd
import numpy as np
import pickle
import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy import sparse

# Configuration 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOG_FILE = "logs/features.log"
os.makedirs("features", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Setup Logging 
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Data Loading 
def load_cleaned_data(file="data/cleaned/tweets_cleaned.parquet") :
    """
    Load cleaned tweet content from a parquet file.

    Args:
        file (str): Path to the cleaned data.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logging.info(f"Loading cleaned data from {file}")
    return pd.read_parquet(file)

# TF-IDF 
def save_sparse_matrix(matrix, filepath):
    """
    Save a sparse matrix to a file.

    Args:
        matrix (scipy.sparse.csr_matrix): Matrix to save.
        filepath (str): Output path.
    """
    sparse.save_npz(filepath, matrix)
    logging.info(f"Saved sparse matrix to {filepath}")

def generate_tfidf_vectors(df: pd.DataFrame):
    """
    Generate and save TF-IDF vectors from tweet content.

    Args:
        df (pd.DataFrame): Cleaned tweets.
    """
    logging.info("Generating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["content"])

    save_sparse_matrix(tfidf_matrix, "features/tfidf_vectors.npz")
    with open("features/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        logging.info("Saved TF-IDF vectorizer to features/tfidf_vectorizer.pkl")

    logging.info(f"TF-IDF shape: {tfidf_matrix.shape}")

# Embeddings 
def generate_sentence_embeddings(df: pd.DataFrame):
    """
    Generate semantic embeddings for tweet content and save them.

    Args:
        df (pd.DataFrame): Cleaned tweets.
    """
    logging.info("Generating sentence embeddings")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = df["content"].apply(lambda x: model.encode(x).tolist())
    df["embedding"] = embeddings
    df.to_parquet("embeddings/tweets_with_embeddings.parquet", index=False)

    logging.info("Sentence embeddings saved to embeddings/tweets_with_embeddings.parquet")

# Custom Features
def generate_custom_features(df: pd.DataFrame):
    """
    Add simple rule-based keyword signal features.

    Args:
        df (pd.DataFrame): Cleaned tweets.
    """
    logging.info("Generating custom keyword signals...")

    buy_keywords = ["buy", "bullish", "long", "breakout", "target"]
    sell_keywords = ["sell", "bearish", "short", "resistance", "fall"]

    def keyword_score(text):
        text = text.lower()
        return sum(w in text for w in buy_keywords) - sum(w in text for w in sell_keywords)

    df["keyword_score"] = df["content"].apply(keyword_score)
    df.to_parquet("features/tweets_with_keywordscore.parquet", index=False)

    logging.info("Saved keyword signals to features/tweets_with_keywordscore.parquet")

# Pipeline 
def main():
    """
    Main execution pipeline:
    - Load cleaned data
    - Generate TF-IDF vectors
    - Generate sentence embeddings
    - Add keyword signal features
    """
    logging.info("Starting feature engineering ")
    df = load_cleaned_data()

    generate_tfidf_vectors(df)
    generate_sentence_embeddings(df)
    generate_custom_features(df)

    logging.info("Feature engineering complete")

if __name__ == "__main__":
    main()
