from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional

from loguru import logger

from llm.news_fetcher import fetch_latest_news
from llm.summarizer import summarize_news


async def build_report(trades: List[dict], metrics: dict, llm_model: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    news = await fetch_latest_news(limit=5)
    news_summary = summarize_news(news, model=llm_model)

    body = [
        f"# BTC Futures Daily Report — {ts}",
        "",
        "## Performance",
        f"- PnL: {metrics.get('pnl', 0):.2f} USDT",
        f"- Sharpe (approx): {metrics.get('sharpe', 0):.3f}",
        f"- Max Drawdown: {metrics.get('max_drawdown', 0):.3f}",
        f"- Win Rate: {metrics.get('win_rate', 0):.3f}",
        "",
        "## Trade Highlights",
    ]
    for t in trades[-5:]:
        body.append(
            f"- ts={t.get('ts')}, action={t.get('action')}, price={t.get('price')}, "
            f"pnl={t.get('pnl', 0):.4f}, equity={t.get('equity', 0):.2f}"
        )

    body.extend(
        [
            "",
            "## Market/Sentiment (LLM)",
            news_summary,
            "",
            "## Raw Headlines",
        ]
    )
    for n in news:
        body.append(f"- {n['title']} ({n.get('link','')})")

    return "\n".join(body)


async def generate(summary: Dict, llm_model: str, period: str = "daily") -> str:
    """
    Generate a daily/weekly report from aggregated summaries.
    """
    news = await fetch_latest_news(limit=5)
    news_summary = summarize_news(news, model=llm_model)
    title = f"BTC Futures {period.capitalize()} Report — {summary.get('date', datetime.utcnow().date())}"
    lines = [
        f"# {title}",
        "",
        "## Performance",
        f"- Return: {summary.get('return_pct', 0):.2%}",
        f"- Max Drawdown: {summary.get('max_drawdown', 0):.3f}",
        f"- Sharpe: {summary.get('sharpe', 0):.3f}",
        f"- Trades: {summary.get('trade_count', 0)}",
        "",
        "## Top Trades",
    ]
    for t in summary.get("top_trades", [])[:3]:
        lines.append(
            f"- ts={t.get('ts')}, action={t.get('action')}, pnl={t.get('pnl', 0):.4f}, entry={t.get('entry')}, exit={t.get('exit')}"
        )
    lines.extend(
        [
            "",
            "## Risk Notes",
            f"- Daily PnL: {summary.get('pnl', 0):.2f}",
            f"- Worst Trade: {summary.get('worst_trade', {})}",
            "",
            "## Market/Sentiment",
            news_summary,
            "",
            "## Raw Headlines",
        ]
    )
    for n in news:
        lines.append(f"- {n['title']} ({n.get('link','')})")
    return "\n".join(lines)
