"""
Scraping package for NBA.com data collection.

Currently provides a Scrapy-Playwright spider that scrapes the NBA.com
players roster page (with "Show Historic" enabled) to extract every
player's headshot image URL from the rendered <img> elements.

Run via:
    python run_scrape.py
"""
