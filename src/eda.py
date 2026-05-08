import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # Parse preserving local timezone so hour extraction is in original local time
    raw = pd.to_datetime(df["date"], format="mixed", utc=False)
    # raw may be object dtype with mixed TZ offsets; .apply is safe in both cases
    df["pub_date"] = pd.to_datetime(
        raw.apply(lambda x: x.date() if pd.notna(x) else None)
    )
    df["pub_hour"] = raw.apply(
        lambda x: x.hour if pd.notna(x) else None
    ).astype("Int64")
    # UTC-normalised datetime kept for consistent time-series resampling
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original = len(df)

    df = df.drop_duplicates()
    after_dedup = len(df)

    missing_mask = df["headline"].isna() | df["publisher"].isna() | df["date"].isna() | df["pub_date"].isna() | df["pub_hour"].isna()
    df = df[~missing_mask].reset_index(drop=True)
    after_missing = len(df)

    report = {
        "original_rows": original,
        "duplicates_removed": original - after_dedup,
        "missing_removed": after_dedup - after_missing,
        "final_rows": after_missing,
        "missing_by_column": {
            col: int(df[col].isna().sum())
            for col in ["headline", "url", "publisher", "date", "stock"]
        },
    }
    return df, report


def headline_length_stats(df: pd.DataFrame) -> pd.Series:
    lengths = df["headline"].str.len()
    stats = lengths.describe().round(2)
    stats["median"] = round(lengths.median(), 2)
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
    return df.set_index("pub_date").resample(freq)["headline"].count()


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
    daily = df.set_index("pub_date").resample("D")["headline"].count()
    top = daily.nlargest(top_n).reset_index()
    top.columns = ["date", "article_count"]
    top["date"] = top["date"].dt.date
    return top


# ── Text Analysis ────────────────────────────────────────────────────────────

FINANCIAL_PHRASES = [
    "price target", "earnings beat", "earnings miss", "earnings per share",
    "FDA approval", "FDA approved", "clinical trial", "drug approval",
    "interest rate", "rate hike", "rate cut", "federal reserve",
    "quarterly earnings", "revenue growth", "revenue beat", "guidance raised",
    "guidance cut", "market cap", "stock buyback", "share repurchase",
    "dividend", "special dividend", "IPO", "merger", "acquisition",
    "upgraded", "downgraded", "buy rating", "sell rating", "hold rating",
    "52 week high", "52 week low", "short squeeze", "insider buying",
]


def top_count_terms(df: pd.DataFrame, top_n: int = 30, ngram_range: tuple = (1, 1)) -> pd.Series:
    vec = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=10000)
    X = vec.fit_transform(df["headline"].dropna())
    counts = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    return pd.Series(counts, index=terms).nlargest(top_n)


def top_tfidf_terms(df: pd.DataFrame, top_n: int = 30, ngram_range: tuple = (1, 1)) -> pd.Series:
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, max_features=5000)
    X = vec.fit_transform(df["headline"].dropna())
    # Mean TF-IDF score per term — high score = frequent AND distinctive
    scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    return pd.Series(scores, index=terms).nlargest(top_n)


def financial_phrase_counts(df: pd.DataFrame,
                            phrases: list[str] = None) -> pd.Series:
    if phrases is None:
        phrases = FINANCIAL_PHRASES
    headlines = df["headline"].dropna().str.lower()
    counts = {p: int(headlines.str.contains(p, regex=False).sum()) for p in phrases}
    return pd.Series(counts).sort_values(ascending=False)


def plot_top_terms(series: pd.Series, title: str = "Top Terms", save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    series.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_financial_phrases(df: pd.DataFrame, save_path: str = None):
    counts = financial_phrase_counts(df)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(12, 7))
    counts.sort_values().plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Financial Phrase Frequency in Headlines")
    ax.set_xlabel("Number of Headlines")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def lda_topics(df: pd.DataFrame, n_topics: int = 8, top_words: int = 10,
               sample_n: int = 100_000) -> list[dict]:
    headlines = df["headline"].dropna()
    if len(headlines) > sample_n:
        # Stratified sample across months so all time periods are represented
        tmp = df.loc[headlines.index, ["headline", "pub_date"]].copy()
        tmp["_period"] = tmp["pub_date"].dt.to_period("M")
        n_periods = tmp["_period"].nunique()
        per_period = max(1, sample_n // n_periods)
        headlines = (
            tmp.groupby("_period", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_period), random_state=42))
               ["headline"]
               .dropna()
        )
        if len(headlines) > sample_n:
            headlines = headlines.sample(sample_n, random_state=42)
    vec = CountVectorizer(stop_words="english", max_features=5000, min_df=5)
    X = vec.fit_transform(headlines)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    max_iter=15, n_jobs=-1)
    lda.fit(X)
    terms = vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top = [terms[j] for j in comp.argsort()[:-top_words - 1:-1]]
        topics.append({"topic": i + 1, "words": top})
    return topics


def plot_lda_topics(topics: list[dict], save_path: str = None):
    n = len(topics)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for ax, t in zip(axes, topics):
        ax.barh(range(len(t["words"])), range(len(t["words"]), 0, -1),
                color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(t["words"])))
        ax.set_yticklabels(t["words"], fontsize=9)
        ax.set_title(f"Topic {t['topic']}", fontsize=10)
        ax.set_xlabel("Rank weight")
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle("LDA Topics — Top Words per Topic", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Time Series – Publishing Hours ───────────────────────────────────────────

def publishing_hour_distribution(df: pd.DataFrame) -> pd.Series:
    return df["pub_hour"].value_counts().sort_index()


def plot_publishing_hours(df: pd.DataFrame, save_path: str = None):
    counts = publishing_hour_distribution(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
    ax.set_title("Article Publication Volume by Hour of Day (local time)")
    ax.set_xlabel("Hour (local time)")
    ax.set_ylabel("Number of Articles")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Publisher Analysis ────────────────────────────────────────────────────────

def extract_publisher_domains(df: pd.DataFrame) -> pd.Series:
    """Return domain counts for publishers that look like email addresses."""
    emails = df["publisher"][df["publisher"].str.contains("@", na=False)]
    domains = emails.str.extract(r"@([\w.\-]+)")[0]
    return domains.value_counts()


def plot_publisher_domains(df: pd.DataFrame, top_n: int = 20, save_path: str = None):
    domains = extract_publisher_domains(df).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    domains.sort_values().plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title(f"Top {top_n} Publisher Domains (email-based publishers)")
    ax.set_xlabel("Number of Articles")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def publisher_coverage_profile(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """For each top publisher: article count, unique stocks covered, date range."""
    top_pubs = df["publisher"].value_counts().head(top_n).index
    rows = []
    for pub in top_pubs:
        sub = df[df["publisher"] == pub]
        rows.append({
            "publisher": pub,
            "articles": len(sub),
            "unique_stocks": sub["stock"].nunique(),
            "first_date": sub["date"].min().date(),
            "last_date": sub["date"].max().date(),
        })
    return pd.DataFrame(rows)
