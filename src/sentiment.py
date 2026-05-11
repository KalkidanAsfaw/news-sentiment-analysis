from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Data loading ──────────────────────────────────────────────────────────────

def load_news(path: str, tickers: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    df["pub_date"] = df["date"].dt.normalize().dt.tz_localize(None).dt.date
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    if tickers is not None:
        df = df[df["stock"].isin(tickers)].copy()
    return df.reset_index(drop=True)


def load_stock(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = "date"
    df["ticker"] = ticker
    return df


# ── Technical: daily returns ──────────────────────────────────────────────────

def compute_daily_returns(stock_df: pd.DataFrame) -> pd.Series:
    """Percentage daily return based on Adj Close (falls back to Close)."""
    price_col = "Adj Close" if "Adj Close" in stock_df.columns else "Close"
    returns = stock_df[price_col].pct_change() * 100
    returns.name = "daily_return"
    return returns


# ── Date alignment ────────────────────────────────────────────────────────────

def align_to_next_trading_day(
    pub_dates: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Map each publication date to the current or next trading day.

    Articles published on weekends or market holidays are pushed forward to the
    next available trading session, matching how a market participant would act
    on the information.
    """
    trading_set = pd.DatetimeIndex(sorted(trading_dates))

    def _next_trading(d):
        if pd.isna(d):
            return pd.NaT
        # searchsorted returns the insertion point; gives the next-or-equal date
        idx = trading_set.searchsorted(d, side="left")
        if idx >= len(trading_set):
            return pd.NaT
        return trading_set[idx]

    return pub_dates.apply(_next_trading)


# ── Sentiment scoring ─────────────────────────────────────────────────────────

_analyzer = SentimentIntensityAnalyzer()


def score_sentiment(headlines: pd.Series) -> pd.Series:
    """Return VADER compound scores in [-1, 1] for each headline."""
    return headlines.apply(
        lambda h: _analyzer.polarity_scores(str(h))["compound"] if pd.notna(h) else np.nan
    )


def sentiment_category(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    ticker: str,
) -> pd.DataFrame:
    """Average VADER compound score per aligned trading day for one ticker."""
    sub = news_df[news_df["stock"] == ticker].copy()
    sub["aligned_date"] = align_to_next_trading_day(sub["pub_date"], trading_dates)
    sub = sub.dropna(subset=["aligned_date", "sentiment"])
    daily = (
        sub.groupby("aligned_date")["sentiment"]
        .agg(sentiment_score="mean", article_count="count")
        .reset_index()
        .rename(columns={"aligned_date": "date"})
    )
    daily["ticker"] = ticker
    return daily


def build_merged_df(
    news_df: pd.DataFrame,
    stock_dfs: dict[str, pd.DataFrame],
    tickers: list[str],
) -> pd.DataFrame:
    """Join daily sentiment and daily returns for all tickers into one frame."""
    parts = []
    for ticker in tickers:
        stock = stock_dfs[ticker].copy()
        returns = compute_daily_returns(stock)
        returns_df = returns.reset_index()
        returns_df.columns = ["date", "daily_return"]
        returns_df["ticker"] = ticker

        trading_dates = pd.DatetimeIndex(stock.index)
        sentiment_df = aggregate_daily_sentiment(news_df, trading_dates, ticker)

        merged = pd.merge(sentiment_df, returns_df, on=["date", "ticker"], how="inner")
        parts.append(merged)

    return pd.concat(parts, ignore_index=True)


# ── Correlation ───────────────────────────────────────────────────────────────

def pearson_per_ticker(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r, p-value, and n per ticker."""
    rows = []
    for ticker, grp in merged_df.groupby("ticker"):
        valid = grp.dropna(subset=["sentiment_score", "daily_return"])
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid["sentiment_score"], valid["daily_return"])
        rows.append({
            "ticker": ticker,
            "pearson_r": round(r, 4),
            "p_value": round(p, 4),
            "n_days": len(valid),
            "significant": p < 0.05,
        })
    if not rows:
        return pd.DataFrame(columns=["ticker", "pearson_r", "p_value", "n_days", "significant"])
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_scatter_grid(
    merged_df: pd.DataFrame,
    tickers: list[str],
    save_path: str | None = None,
):
    """Scatter plot of daily sentiment vs daily return, one panel per ticker."""
    n = len(tickers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, ticker in zip(axes, tickers):
        grp = merged_df[merged_df["ticker"] == ticker].dropna(
            subset=["sentiment_score", "daily_return"]
        )
        if len(grp) < 5:
            ax.set_visible(False)
            continue
        r, p = stats.pearsonr(grp["sentiment_score"], grp["daily_return"])
        ax.scatter(
            grp["sentiment_score"], grp["daily_return"],
            alpha=0.35, s=18, color="steelblue", edgecolors="none",
        )
        # Regression line
        m, b = np.polyfit(grp["sentiment_score"], grp["daily_return"], 1)
        xs = np.linspace(grp["sentiment_score"].min(), grp["sentiment_score"].max(), 100)
        ax.plot(xs, m * xs + b, color="tomato", linewidth=1.4)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"{ticker}  r = {r:.3f}{sig}  (n={len(grp)})", fontsize=11)
        ax.set_xlabel("Avg Daily Sentiment (VADER compound)")
        ax.set_ylabel("Daily Return (%)")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "News Sentiment vs Daily Stock Returns\n* p < 0.05",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_return_by_sentiment_category(
    merged_df: pd.DataFrame,
    save_path: str | None = None,
):
    """Bar chart: average daily return for positive / neutral / negative days."""
    df = merged_df.dropna(subset=["sentiment_score", "daily_return"]).copy()
    df["category"] = df["sentiment_score"].apply(sentiment_category)

    order = ["positive", "neutral", "negative"]
    palette = {"positive": "#2ca02c", "neutral": "#aec7e8", "negative": "#d62728"}

    fig, axes = plt.subplots(1, len(df["ticker"].unique()), figsize=(5 * df["ticker"].nunique(), 5))
    if df["ticker"].nunique() == 1:
        axes = [axes]

    for ax, ticker in zip(np.array(axes).flatten(), sorted(df["ticker"].unique())):
        grp = df[df["ticker"] == ticker]
        means = grp.groupby("category")["daily_return"].mean().reindex(order)
        counts = grp["category"].value_counts().reindex(order, fill_value=0)
        colors = [palette[c] for c in order]
        bars = ax.bar(order, means.values, color=colors, edgecolor="white", width=0.6)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * (1 if bar.get_height() >= 0 else -1),
                f"n={cnt}", ha="center", va="bottom", fontsize=9,
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(ticker, fontsize=11)
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Avg Daily Return (%)")

    plt.suptitle("Average Daily Return by News Sentiment Category", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sentiment_timeline(
    news_df: pd.DataFrame,
    stock_dfs: dict[str, pd.DataFrame],
    ticker: str,
    save_path: str | None = None,
):
    """Rolling 30-day average sentiment alongside closing price for one ticker."""
    trading_dates = pd.DatetimeIndex(stock_dfs[ticker].index)
    daily_sent = aggregate_daily_sentiment(news_df, trading_dates, ticker)
    daily_sent = daily_sent.set_index("date").sort_index()

    price_col = "Adj Close" if "Adj Close" in stock_dfs[ticker].columns else "Close"
    price = stock_dfs[ticker][price_col]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(price.index, price.values, color="steelblue", linewidth=1.2, label="Close")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{ticker} — Closing Price and 30-Day Rolling Sentiment")
    ax1.legend(loc="upper left")

    roll = daily_sent["sentiment_score"].rolling(30, min_periods=5).mean()
    ax2.plot(daily_sent.index, daily_sent["sentiment_score"], alpha=0.25, color="gray", linewidth=0.8)
    ax2.plot(roll.index, roll.values, color="darkorange", linewidth=1.4, label="30-day rolling avg")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.fill_between(roll.index, roll.values, 0,
                     where=(roll > 0), alpha=0.15, color="green")
    ax2.fill_between(roll.index, roll.values, 0,
                     where=(roll < 0), alpha=0.15, color="red")
    ax2.set_ylabel("VADER Compound Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
