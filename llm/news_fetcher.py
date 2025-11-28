from __future__ import annotations

import httpx
from loguru import logger


async def fetch_latest_news(limit: int = 5) -> list[dict]:
    """
    Placeholder news fetcher using CoinDesk RSS-to-JSON mirror.
    """
    url = "https://api.rss2json.com/v1/api.json?rss_url=https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])[:limit]
            return [{"title": i["title"], "summary": i.get("description", ""), "link": i.get("link", "")} for i in items]
    except Exception as exc:
        logger.error(f"Failed to fetch news: {exc}")
        return []
