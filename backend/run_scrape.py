#!/usr/bin/env python3
"""
Entry-point script to run the NBA player headshot scraping spider.

Usage:
    cd backend
    python run_scrape.py

This launches the Scrapy-Playwright spider that scrapes
https://www.nba.com/players (with "Show Historic" enabled) and writes
the headshot URL mapping to backend/data/nba_player_headshots.json.

Prerequisites:
    pip install -r requirements.txt
    playwright install chromium
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from app.scraping.settings import (
    DOWNLOAD_HANDLERS,
    DOWNLOAD_DELAY,
    CONCURRENT_REQUESTS,
    PLAYWRIGHT_BROWSER_TYPE,
    PLAYWRIGHT_LAUNCH_OPTIONS,
    PLAYWRIGHT_CONTEXTS_DEFAULTS,
    PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT,
    USER_AGENT,
    FEED_EXPORT_ENCODING,
    LOG_LEVEL,
    ROBOTSTXT_OBEY,
    TWISTED_REACTOR,
)
from app.scraping.pipelines import HeadshotPipeline


from app.scraping.spiders.nba_players_spider import NbaPlayersHeadshotSpider


def main() -> None:
    settings = {
        "DOWNLOAD_HANDLERS": DOWNLOAD_HANDLERS,
        "TWISTED_REACTOR": TWISTED_REACTOR,
        "PLAYWRIGHT_BROWSER_TYPE": PLAYWRIGHT_BROWSER_TYPE,
        "PLAYWRIGHT_LAUNCH_OPTIONS": PLAYWRIGHT_LAUNCH_OPTIONS,
        "PLAYWRIGHT_CONTEXTS_DEFAULTS": PLAYWRIGHT_CONTEXTS_DEFAULTS,
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT,
        "PLAYWRIGHT_REQUESTS": True,
        "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
        "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
        "USER_AGENT": USER_AGENT,
        "FEED_EXPORT_ENCODING": FEED_EXPORT_ENCODING,
        "LOG_LEVEL": LOG_LEVEL,
        "ROBOTSTXT_OBEY": ROBOTSTXT_OBEY,
        "ITEM_PIPELINES": {
            "app.scraping.pipelines.HeadshotPipeline": 300,
        },
    }

    process = CrawlerProcess(settings)
    process.crawl(NbaPlayersHeadshotSpider)
    process.start()


if __name__ == "__main__":
    main()
