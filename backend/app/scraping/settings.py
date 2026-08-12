"""
Scrapy settings for the NBA player headshot spider.

Uses scrapy-playwright to render the JavaScript-driven nba.com/players
page (a Next.js client-side rendered app). A real Chromium browser is
required because the player table is populated by JS after page load.
"""

from __future__ import annotations

# ── Playwright integration ──
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": False,
    "args": [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
    ],
}

# Realistic browser context settings to avoid bot detection
PLAYWRIGHT_CONTEXTS_DEFAULTS = {
    "viewport": {"width": 1920, "height": 1080},
    "user_agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "locale": "en-US",
    "timezone_id": "America/New_York",
}

# ── Politeness ──
DOWNLOAD_DELAY = 1.0
CONCURRENT_REQUESTS = 2
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30_000  # 30s — nba.com is slow

# ── User agent (real browser UA to avoid bot detection) ──
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── Output ──
FEED_EXPORT_ENCODING = "utf-8"
LOG_LEVEL = "INFO"

# ── Robots ──
ROBOTSTXT_OBEY = False  # nba.com robots.txt blocks /players; we scrape respectfully
