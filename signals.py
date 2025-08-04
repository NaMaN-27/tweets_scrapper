"""
Signal Aggregation Script

This script classifies tweet sentiment using keyword-based rules,
then aggregates the results by day into trading signals with confidence scores.

Author: Your Name
Date: 2025-08-03
"""

import pandas as pd
import os
import logging

# Logging 

os.makedirs("signals", exist_ok=True)

logging.basicConfig(
    filename="logs/signals.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Signal Rules
buy_threshold = 0.65
sell_threshold = 0.35

def classify_sentiment(text) :
    """
    Classify sentiment of tweet content as 'buy', 'sell', or 'neutral'.

    Args:
        text (str): Tweet text.

    Returns:
        str: Sentiment category.
    """
    buy_keywords = ["buy", "bullish", "long", "breakout", "target", "support"]
    sell_keywords = ["sell", "bearish", "short", "resistance", "fall", "downside"]

    text = text.lower()
    buy_score = sum(word in text for word in buy_keywords)
    sell_score = sum(word in text for word in sell_keywords)

    if buy_score > sell_score:
        return "buy"
    elif sell_score > buy_score:
        return "sell"
    else:
        return "neutral"

def compute_aggregated_signals(group) :
    """
    Aggregate tweet-level signals into daily metrics and compute a composite signal.

    Args:
        group (pd.DataFrame): Tweets for one day.

    Returns:
        pd.Series: Aggregated metrics and final signal.
    """
    total = len(group)
    if total == 0:
        return pd.Series({
            "tweet_volume": 0,
            "buy_pct": 0,
            "sell_pct": 0,
            "neutral_pct": 0,
            "avg_keyword_score": 0,
            "composite_score": 0,
            "signal": "neutral",
            "confidence_pct": 0
        })

    buy = (group['sentiment'] == 'buy').sum()
    sell = (group['sentiment'] == 'sell').sum()
    neutral = (group['sentiment'] == 'neutral').sum()

    buy_pct = buy / total
    sell_pct = sell / total
    neutral_pct = neutral / total
    avg_keyword = group["keyword_score"].mean()

    # Weighted signal scoring
    score = 0.5 * buy_pct + 0.3 * (avg_keyword / 5) + 0.2 * min(total / 500, 1.0)
    signal = "buy" if score > buy_threshold else "sell" if score < sell_threshold else "neutral"
    confidence = round(score * 100, 1)

    return pd.Series({
        "tweet_volume": total,
        "buy_pct": round(buy_pct, 3),
        "sell_pct": round(sell_pct, 3),
        "neutral_pct": round(neutral_pct, 3),
        "avg_keyword_score": round(avg_keyword, 2),
        "composite_score": round(score, 3),
        "signal": signal,
        "confidence_pct": confidence
    })

# Aggregation  
def main():
    """
    Load tweets with keyword scores, classify sentiment, and aggregate daily signals.
    Saves the result to a Parquet file and logs the pipeline.
    """
    input_file = "features/tweets_with_keywordscore.parquet"
    output_file = "signals/daily_aggregated_signals.csv"

    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        print(f"File missing: {input_file}")
        return

    logging.info(f"Loading: {input_file}")
    df = pd.read_parquet(input_file)

    logging.info(f"Classifying sentiment for {len(df)} tweets")
    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["date"] = df["datetime"].dt.date
    df["sentiment"] = df["content"].apply(classify_sentiment)

    logging.info("Aggregating by date")
    aggregated = df.groupby("date").apply(compute_aggregated_signals).reset_index()

    logging.info(f"Saving aggregated output to {output_file}")
    aggregated.to_csv(output_file, index=False)

    print(f"Saved {len(aggregated)} daily signals to: {output_file}")
    logging.info("Signal aggregation complete.")

if __name__ == "__main__":
    main()
