# Twitter-Based Trading Signal Analyzer

This project collects, cleans, analyzes, and visualizes Twitter data (with a focus on Indian stock market hashtags like `#nifty50`, `#banknifty`, `#sensex`) to generate sentiment-based trading signals using NLP.

## 📂 Folder Structure

```text
.
├── data/
│   ├── raw/             # Raw tweet data (parquet)
│   ├── cleaned/         # Cleaned tweet data
│   └── features/        # TF-IDF, embeddings, keyword scores
├── embeddings/          # Sentence embeddings
├── features/            # TF-IDF vectors, keyword scores
├── logs/                # Pipeline logs
├── visualizations/      # Output plots
├── signals/             # Daily aggregated signal files
├── clean_tweets.py
├── feature_eigineering.py
├── signals.py
├── visualize.py
├── twitter_login.py
├── scraper.py
├── app.py               # Streamlit interface
├── requirements.txt
└── README.md
```


## 🔄 Pipeline Overview

1. **Scrape Tweets**  
   → Use `playwright` with logged-in session to extract tweets containing finance-related hashtags.

2. **Clean Tweets**  
   → Normalize text, handle emojis/Unicode, detect language, deduplicate.

3. **Feature Engineering**  
   → Generate TF-IDF vectors, sentence embeddings, and domain-specific keyword scores.

4. **Signal Aggregation**  
   → Group by date and classify `buy`, `sell`, or `neutral` using heuristics + confidence score.

5. **Visualization**  
   → PCA-based plots

6. **Streamlit App**  
   → One-click interface to run each stage and view results.

---

## 🧪 Example Signal Output

| date       | tweet_volume | buy_pct | sell_pct | signal  | confidence_pct |
|------------|--------------|---------|----------|---------|----------------|
| 2025-08-03 | 1523         | 0.61    | 0.24     | buy     | 74.2           |
| 2025-08-04 | 1487         | 0.32    | 0.45     | sell    | 67.5           |

---

## 🚀 How to Run

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/twitter-trading-signals.git
cd twitter-trading-signals
pip install -r requirements.txt

# 2. Place your logged-in Twitter state
To genrate this json run twitter_login.py and login to twitter account

# 3. Run the scraper (optional step)
this can be also automated using a scheduled job which will trigger this every morning before market opens 
python scraper.py

# 4. Launch Streamlit dashboard
streamlit run app.py
