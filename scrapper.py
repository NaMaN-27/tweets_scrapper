import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re
import hashlib
import random
import logging
import os
from datetime import datetime

# Setup Logging 
os.makedirs("logs", exist_ok=True)
log_file = f"logs/run.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Utility Functions 

def extract_entities(text):
    """Extract mentions and hashtags from tweet content."""
    mentions = re.findall(r"@\w+", text)
    hashtags = re.findall(r"#\w+", text)
    return mentions, hashtags

def parse_count(text):
    """Convert likes/retweet text (e.g., '1.2K') to int."""
    try:
        text = text.replace(",", "").strip()
        if "K" in text:
            return int(float(text.replace("K", "")) * 1000)
        elif "M" in text:
            return int(float(text.replace("M", "")) * 1_000_000)
        return int(text)
    except:
        return 0

def get_tweet_hash(username, timestamp, content):
    """Hash a tweet using username, timestamp, and content."""
    base = f"{username}|{timestamp}|{content.strip()}"
    return hashlib.md5(base.encode()).hexdigest()

# Core Scraping

async def scroll_and_collect_tweets(page, tag, seen_hashes, max_tweets=2000, max_scrolls=150):
    """
    Scroll through Twitter search results and collect tweets.

    Args:
        page: Playwright page instance
        tag: Hashtag string (e.g., "#nifty50")
        seen_hashes: Set of tweet hashes to avoid duplicates
        max_tweets: Max tweets to collect per tag
        max_scrolls: Max scroll iterations

    Returns:
        List of tweet dictionaries
    """
    tweets_data = []

    for scroll_num in range(max_scrolls):
        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(1500 + random.randint(200, 600))

        tweet_blocks = await page.query_selector_all("article:has(time)")
        logging.info(f"{tag} | Scroll {scroll_num+1}: {len(tweet_blocks)} tweet containers")

        for block in tweet_blocks:
            try:
                content = await block.inner_text()
                if not content.strip():
                    continue

                time_tag = await block.query_selector("time")
                timestamp = await time_tag.get_attribute("datetime") if time_tag else "N/A"

                user_tag = await block.query_selector('div[data-testid="User-Name"] span')
                username = await user_tag.inner_text() if user_tag else "N/A"

                like_span = await block.query_selector('div[data-testid="like"] span')
                likes = parse_count(await like_span.inner_text()) if like_span else 0

                retweet_span = await block.query_selector('div[data-testid="retweet"] span')
                retweets = parse_count(await retweet_span.inner_text()) if retweet_span else 0

                mentions, hashtags = extract_entities(content)
                tweet_hash = get_tweet_hash(username, timestamp, content)

                if tweet_hash in seen_hashes:
                    continue
                seen_hashes.add(tweet_hash)

                tweets_data.append({
                    "hashtag": tag,
                    "username": username,
                    "timestamp": timestamp,
                    "content": content.replace("\n", " "),
                    "likes": likes,
                    "retweets": retweets,
                    "mentions": ", ".join(mentions),
                    "hashtags": ", ".join(hashtags)
                })

                if len(tweets_data) % 100 == 0:
                    logging.info(f"{tag}: {len(tweets_data)} tweets collected")

                if len(tweets_data) >= max_tweets:
                    logging.info(f"{tag}: Reached max tweets limit ({max_tweets})")
                    return tweets_data

            except Exception as e:
                logging.warning(f"{tag}: Skipped tweet due to error: {e}")
                continue

    return tweets_data

async def scrape_multiple_tags(storage_file="twitter_storage.json", tags=["#nifty50", "#banknifty"], max_tweets=2000, max_scrolls=150):
    """
    Scrape multiple Twitter hashtags and save results.

    Args:
        storage_file: Logged-in browser storage state file
        tags: List of hashtags to scrape
        max_tweets: Max tweets per tag
        max_scrolls: Max scroll iterations per tag
    """
    all_data = []
    seen_hashes = set()
    os.makedirs("data", exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=storage_file)
        page = await context.new_page()

        for tag in tags:
            try:
                logging.info(f"Searching: {tag}")
                query = tag.replace("#", "%23")
                url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"

                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(5000)

                tag_data = await scroll_and_collect_tweets(page, tag, seen_hashes, max_tweets, max_scrolls)
                all_data.extend(tag_data)
            except Exception as e:
                logging.error(f"Error scraping {tag}: {e}")

        await browser.close()

    if not all_data:
        logging.error("No tweets found.")
        print("No tweets found across all tags.")
    else:
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = f"data/raw/tweets_{timestamp}.csv"
        parquet_path = f"data/raw/tweets_{timestamp}.parquet"

        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        logging.info(f"Saved {len(df)} tweets to CSV and Parquet.")
        print(f"\n Saved {len(df)} tweets to:\n- {csv_path}\n- {parquet_path}")

if __name__ == "__main__":
    asyncio.run(scrape_multiple_tags())
