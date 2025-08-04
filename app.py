import streamlit as st
import os
import subprocess
import pandas as pd

st.set_page_config(page_title="Twitter Trading Signal Analyzer", layout="wide")

st.title("Twitter-Based Trading Signal App")

# File Path 
raw_path = "data/raw/tweets.parquet"

if os.path.exists(raw_path):
    st.success("Found raw tweet data in 'data/raw/'")

    #Run Cleaner 
    if st.sidebar.button("Run Cleaning"):
        with st.spinner("Cleaning and normalizing tweets..."):
            subprocess.run(["python", "clean_tweets.py", raw_path])
        st.success("Tweets cleaned and saved to 'tweets_cleaned.parquet'")

    # Visualize 
    if st.sidebar.button("Visualize Vectors"):
        with st.spinner("Visualizing TF-IDF and embeddings..."):
            subprocess.run(["python", "visualize.py"])
        st.image("visualizations/tfidf_pca_plot.png", caption="TF-IDF PCA", use_container_width=True)
        st.image("visualizations/embedding_pca_plot.png", caption="Embeddings PCA", use_container_width=True)

    # Feature Engineering 
    if st.sidebar.button("Generate Features"):
        with st.spinner("Running TF-IDF, embeddings, and custom signals..."):
            subprocess.run(["python", "feature_eigineering.py"])
        st.success("✅ Features generated.")

    #  Signal Aggregation 
    if st.sidebar.button("Aggregate Signals"):
        with st.spinner("Classifying sentiment and computing composite signals..."):
            subprocess.run(["python", "signals.py"])

        signal_df = None
        try:
            signal_df = pd.read_csv("signals/daily_aggregated_signals.csv")
            st.success("Signals computed and saved.")
            st.subheader("Daily Signals")
            st.dataframe(signal_df)
        except Exception as e:
            st.error(f"Failed to load signals: {e}")
else:
    st.warning("Please ensure 'data/raw/tweets.parquet' exists.")
