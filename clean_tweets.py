import pandas as pd
import re
import unicodedata
import emoji
from langdetect import detect, DetectorFactory
from datetime import datetime
import pytz
import os
import logging
import sys

# Setup 
DetectorFactory.seed = 42
os.makedirs("data/cleaned", exist_ok=True)

logging.basicConfig(
    filename="logs/cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cleaning Functions 
def normalize_unicode(text):
    """
    Normalize invisible Unicode characters and apply NFKC normalization.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', '', text)  # invisible Unicode
    return text

def remove_emojis(text):
    """
    Remove all emojis from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without emojis.
    """
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    """
    Clean text by normalizing, removing newlines, emojis, and extra whitespace.

    Args:
        text (str): Raw tweet content.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = normalize_unicode(text)
    text = text.replace("\n", " ").replace("\r", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_emojis(text)
    return text

def clean_mentions_or_hashtags(field):
    """
    Clean and deduplicate mentions or hashtags.

    Args:
        field (str): Comma-separated mentions or hashtags.

    Returns:
        str: Cleaned and normalized string.
    """
    if pd.isna(field):
        return ""
    parts = re.split(r",\s*", field)
    cleaned = [re.sub(r"[^\w@#]", "", p.strip().lower()) for p in parts if p]
    return ", ".join(sorted(set(cleaned)))

def parse_timestamp(ts):
    """
    Convert UTC ISO timestamp to Asia/Kolkata timezone ISO format.

    Args:
        ts (str): UTC timestamp.

    Returns:
        str: IST timestamp or 'N/A' on failure.
    """
    try:
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert("Asia/Kolkata").isoformat()
    except Exception as e:
        return "N/A"

def is_valid_language(text, allowed=["en", "hi"]):
    """
    Check if a tweet is in an allowed language (default: English or Hindi).

    Args:
        text (str): Tweet content.

    Returns:
        bool: True if language is allowed, False otherwise.
    """
    try:
        lang = detect(text)
        return lang in allowed
    except:
        return False

# DataFrame Cleaner 
def clean_dataframe(df):
    """
    Clean an entire DataFrame of tweets:
    - Text normalization
    - Language filtering
    - Timestamp conversion
    - Deduplication

    Args:
        df (pd.DataFrame): Raw tweet DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info("Cleaning DataFrame...")

    # Clean text fields
    for col in ["username", "timestamp", "content", "mentions", "hashtags"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
        else:
            logging.warning(f"Missing column: {col}")

    df["mentions"] = df["mentions"].apply(clean_mentions_or_hashtags)
    df["hashtags"] = df["hashtags"].apply(clean_mentions_or_hashtags)

    df["timestamp"] = df["timestamp"].apply(parse_timestamp)

    before = len(df)
    df = df[df["content"].str.len() > 5]
    df = df[df["username"] != "N/A"]
    df = df[df["content"].apply(is_valid_language)]
    df = df.drop_duplicates(subset=["username", "timestamp", "content"])
    after = len(df)

    logging.info(f"Cleaned from {before} to {after} rows.")
    return df

# Main 
def clean():
    """
    Load raw tweet data from `data/raw/tweets.parquet`,
    clean it, and save the result to `data/cleaned/tweets_cleaned.parquet`.
    Logs all steps to `logs/cleaning.log`.
    """
    input_file = "data/raw/tweets.parquet"
    output_file = "data/cleaned/tweets_cleaned.parquet"

    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    logging.info(f"Loading: {input_file}")
    print(f"Loading: {input_file}")
    df = pd.read_parquet(input_file)

    logging.info(f"Cleaning {len(df)} rows...")
    df_cleaned = clean_dataframe(df)

    logging.info(f"Saving cleaned data to {output_file}")
    df_cleaned.to_parquet(output_file, index=False, engine="pyarrow")

    print(f"Cleaned {len(df_cleaned)} rows saved to: {output_file}")
    logging.info("Cleaning completed.")

if __name__ == "__main__":
    clean()
