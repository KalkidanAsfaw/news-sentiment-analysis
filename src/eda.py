import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=["date"], infer_datetime_format=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def headline_length_stats(df: pd.DataFrame) -> pd.Series:
    lengths = df["headline"].str.len()
    stats = lengths.describe()
    stats["median"] = lengths.median()
    return stats


def plot_headline_length_distribution(df: pd.DataFrame, save_path: str = None):
    lengths = df["headline"].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("Headline Character Count Distribution")
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(lengths, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7))
    axes[1].set_title("Headline Length — Box Plot")
    axes[1].set_xlabel("Character Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def articles_per_publisher(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    return df["publisher"].value_counts().head(top_n)


def plot_articles_per_publisher(df: pd.DataFrame, top_n: int = 20, save_path: str = None):
    counts = articles_per_publisher(df, top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} Most Active Publishers")
    ax.set_xlabel("Number of Articles")
    ax.set_ylabel("Publisher")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


_FREQ_ALIAS = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def publication_date_trends(df: pd.DataFrame, freq: str = "W") -> pd.Series:
    freq = _FREQ_ALIAS.get(freq, freq)
    return df.set_index("date").resample(freq)["headline"].count()


def plot_publication_trends(df: pd.DataFrame, freq: str = "W", save_path: str = None):
    trend = publication_date_trends(df, freq)
    freq_label = {"D": "Daily", "W": "Weekly", "ME": "Monthly", "M": "Monthly"}.get(freq, freq)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(trend.index, trend.values, linewidth=1, color="steelblue")
    ax.fill_between(trend.index, trend.values, alpha=0.2, color="steelblue")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_title(f"{freq_label} Article Publication Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def top_publishing_days(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    daily = df.set_index("date").resample("D")["headline"].count()
    top = daily.nlargest(top_n).reset_index()
    top.columns = ["date", "article_count"]
    top["date"] = top["date"].dt.date
    return top
