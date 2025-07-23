# sentiment/news_fetcher.py

import os
import json
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from config.paths import NEWS_API_KEY_PATH


with open(NEWS_API_KEY_PATH, "r") as f:
    NEWS_API_KEY = f.read().strip()

newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def fetch_today_articles():
    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)

    all_articles = newsapi.get_everything(
        q="Nvidia",
        language="en",
        sort_by="publishedAt",
        from_param=yesterday.strftime("%Y-%m-%d"),
        to=today.strftime("%Y-%m-%d"),
        page_size=20
    )

    articles = all_articles.get("articles", [])

    simplified = []
    for art in articles:
        simplified.append({
            "date": today.strftime("%Y-%m-%d"),
            "title": art.get("title", ""),
            "description": art.get("description", "")
        })

    return simplified


def get_recent_news_with_cache(cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    if today_str not in cache:
        print("Fetching today's news articles...")
        todays_articles = fetch_today_articles()
        cache[today_str] = todays_articles

        # Keep only latest 5 days
        sorted_dates = sorted(cache.keys(), reverse=True)[:5]
        cache = {d: cache[d] for d in sorted_dates}

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    else:
        print("Today's articles already in cache.")

    all_articles = []
    for date in sorted(cache.keys()):
        all_articles.extend(cache[date])

    return all_articles