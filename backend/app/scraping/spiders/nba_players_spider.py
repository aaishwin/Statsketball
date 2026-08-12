"""
Scrapy-Playwright spider for NBA.com player headshots.

Scrapes https://www.nba.com/players with "Show Historic" toggled ON,
extracting every player's headshot <img src> URL and name from the
rendered roster table.

The page is a Next.js client-side rendered app — the table rows do not
exist in the raw HTML. Scrapy-Playwright launches a real Chromium browser
to execute the JavaScript, toggle the historic filter, and extract the
image elements.

Pagination strategy:
  The page shows 50 rows per page across 105 pages (5,204 total historic
  players). The spider iterates through all pages, extracting rows from
  each rendered page.

Output (via pipeline):
  backend/data/nba_player_headshots.json — {player_name: headshot_url}
"""

from __future__ import annotations

import logging
from typing import Any

import scrapy
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Response
from scrapy_playwright.page import PageMethod

from ..items import NbaPlayerHeadshotItem

logger = logging.getLogger(__name__)

BASE_URL = "https://www.nba.com/players"

# CSS selectors for the rendered page (verified against the live DOM)
HISTORIC_TOGGLE_SELECTOR = 'input[name="showHistoric"]'
TABLE_ROW_SELECTOR = "table tbody tr"
PLAYER_IMG_SELECTOR = "table tbody tr img"
NEXT_BUTTON_SELECTOR = 'button[aria-label*="Next"]'


class NbaPlayersHeadshotSpider(scrapy.Spider):
    """Scrape NBA.com/players for player headshot image URLs."""

    name = "nba_player_headshots"
    custom_settings = {
        "PLAYWRIGHT_REQUESTS": True,
    }

    async def start(self):
        """Yield the initial request with Playwright rendering enabled.

        Scrapy 2.13+ uses ``start()`` (async) instead of ``start_requests()``.
        """
        yield scrapy.Request(
            BASE_URL,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_methods": [
                    # Wait for the page to load (nba.com is heavy with ads)
                    PageMethod("wait_for_load_state", "domcontentloaded", timeout=30000),
                ],
            },
            dont_filter=True,
            errback=self.errback,
        )

    async def parse(self, response: Response, **kwargs: Any):
        """
        Parse the rendered players page.

        Toggles "Show Historic" on, then iterates through all pages,
        extracting player name + headshot URL from each row.
        """
        page = response.meta["playwright_page"]

        try:
            # ── Wait for the table to render (nba.com is JS-heavy) ──
            logger.info("Waiting for player table to render…")
            await page.wait_for_selector(TABLE_ROW_SELECTOR, timeout=30000)
            logger.info("Table rendered.")

            # ── Toggle "Show Historic" ON ──
            await self._toggle_historic(page)

            # ── Set page size to maximum (50) and wait for re-render ──
            await self._set_page_size(page)

            # ── Iterate through all pages ──
            page_num = 1
            total_extracted = 0

            while True:
                # Wait for rows to be present on the current page
                await page.wait_for_selector(TABLE_ROW_SELECTOR, timeout=15000)
                # Small delay for images to load their src
                await page.wait_for_timeout(1000)

                items = await self._extract_rows(page)
                total_extracted += len(items)

                for item in items:
                    yield item

                logger.info(
                    "Page %d: extracted %d players (running total: %d)",
                    page_num,
                    len(items),
                    total_extracted,
                )

                # ── Try to go to the next page ──
                has_next = await self._go_to_next_page(page)
                if not has_next:
                    logger.info("No more pages. Total extracted: %d", total_extracted)
                    break

                page_num += 1

        finally:
            await page.close()

    async def _toggle_historic(self, page) -> None:
        """Click the 'Show Historic' toggle to reveal all historic players."""
        logger.info("Toggling 'Show Historic' ON…")

        # Scroll the toggle into view, then click via mouse for React compatibility
        toggle = page.locator(HISTORIC_TOGGLE_SELECTOR)
        await toggle.scroll_into_view_if_needed()
        box = await toggle.bounding_box()
        if box:
            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        else:
            await toggle.check(force=True)

        # Wait for the row count to increase (585 → 5000+)
        await page.wait_for_timeout(3000)
        logger.info("Show Historic toggled ON")

    async def _set_page_size(self, page) -> None:
        """Set the page-size combobox to the maximum value (50 rows/page)."""
        # The page-size combobox is the one near "Page" text
        # Try to find a select with options like "All", "1", "2"...
        try:
            selects = page.locator("select")
            count = await selects.count()
            for i in range(count):
                sel = selects.nth(i)
                # Check if this select has a "50" option or "All" option
                options_text = await sel.evaluate(
                    "el => Array.from(el.options).map(o => o.value + ':' + o.text)"
                )
                if any("50" in t or "All" in t for t in options_text):
                    # Select the largest numeric option or "All"
                    best = None
                    for t in options_text:
                        val, text = t.split(":", 1)
                        if text.strip().lower() == "all":
                            best = val
                            break
                        if val.isdigit() and (best is None or int(val) > int(best)):
                            best = val
                    if best:
                        await sel.select_option(value=best)
                        await page.wait_for_timeout(2000)
                        logger.info("Set page size to %s", best)
                        return
            logger.info("No page-size selector found; using default 50/page")
        except Exception as exc:
            logger.warning("Could not set page size: %s — using default", exc)

    async def _extract_rows(self, page) -> list[NbaPlayerHeadshotItem]:
        """Extract player name + headshot URL from all rendered table rows."""
        results = await page.evaluate(
            """() => {
                const imgs = Array.from(document.querySelectorAll('table tbody tr img'));
                return imgs.map(img => {
                    const alt = img.alt || '';
                    // alt format: "LeBron James Headshot" → strip " Headshot"
                    const name = alt.replace(/\\s+Headshot$/i, '').trim();
                    const src = img.src || '';
                    return {name, src};
                }).filter(item => item.name && item.src);
            }"""
        )

        items: list[NbaPlayerHeadshotItem] = []
        for entry in results:
            item = NbaPlayerHeadshotItem()
            item["name"] = entry["name"]
            item["headshot_url"] = entry["src"]
            items.append(item)

        return items

    async def _go_to_next_page(self, page) -> bool:
        """Click the 'Next Page' button. Returns True if navigation succeeded."""
        try:
            next_btn = page.locator(NEXT_BUTTON_SELECTOR)
            is_disabled = await next_btn.is_disabled()
            if is_disabled:
                return False

            await next_btn.click()
            # Wait for the table to refresh (new rows load)
            await page.wait_for_timeout(1500)
            return True
        except Exception as exc:
            logger.debug("Next page navigation failed: %s", exc)
            return False

    def errback(self, failure):
        """Handle Playwright request failures."""
        logger.error("Playwright request failed: %s", failure)
        if failure.check(IgnoreRequest):
            logger.error("Request was ignored/filtered")
        else:
            logger.error("Failure type: %s", failure.type)
            logger.error("Failure value: %s", failure.value)
