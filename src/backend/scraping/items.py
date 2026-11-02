"""
Scrapy item definitions for NBA player headshot scraping.
"""

from __future__ import annotations

import scrapy


class NbaPlayerHeadshotItem(scrapy.Item):
    """One player's name and headshot CDN URL.

    The ``name`` is extracted from the ``<img alt="... Headshot">`` attribute
    (with the " Headshot" suffix stripped). The ``headshot_url`` is the
    ``<img src>`` attribute, upgraded from the 260x190 thumbnail to the
    1040x760 resolution in the pipeline.
    """

    name = scrapy.Field()
    headshot_url = scrapy.Field()
