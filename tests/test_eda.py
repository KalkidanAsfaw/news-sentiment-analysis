import pandas as pd
import numpy as np
import pytest
from src.eda import (
    headline_length_stats,
    articles_per_publisher,
    publication_date_trends,
    top_publishing_days,
)


@pytest.fixture
def sample_df():
    raw = pd.to_datetime([
        "2021-01-04 10:00:00-04:00", "2021-01-04 14:30:00-04:00",
        "2021-03-15 09:15:00-04:00", "2021-03-15 16:45:00-04:00",
        "2021-06-01 08:00:00-04:00",
    ], format="mixed", utc=False)   # DatetimeIndex, TZ-aware
    return pd.DataFrame({
        "headline": [
            "Short headline",
            "A much longer headline about stock market movements today",
            "Brief",
            "Another medium length headline here",
            "Short",
        ],
        "publisher": ["Reuters", "Bloomberg", "Reuters", "CNBC", "Bloomberg"],
        "date": raw.tz_convert("UTC"),
        "pub_date": pd.to_datetime([t.date() for t in raw]),
        "pub_hour": pd.array([t.hour for t in raw], dtype="Int64"),
        "stock": ["AAPL", "TSLA", "AAPL", "GOOG", "TSLA"],
    })


class TestHeadlineLengthStats:
    def test_returns_series(self, sample_df):
        result = headline_length_stats(sample_df)
        assert isinstance(result, pd.Series)

    def test_includes_standard_stats(self, sample_df):
        result = headline_length_stats(sample_df)
        for key in ["mean", "std", "min", "max", "median"]:
            assert key in result.index

    def test_correct_min_max(self, sample_df):
        result = headline_length_stats(sample_df)
        assert result["min"] == len("Brief")
        assert result["max"] == len("A much longer headline about stock market movements today")

    def test_count_matches_rows(self, sample_df):
        result = headline_length_stats(sample_df)
        assert result["count"] == len(sample_df)


class TestArticlesPerPublisher:
    def test_returns_series(self, sample_df):
        result = articles_per_publisher(sample_df)
        assert isinstance(result, pd.Series)

    def test_reuters_count(self, sample_df):
        result = articles_per_publisher(sample_df)
        assert result["Reuters"] == 2

    def test_top_n_limit(self, sample_df):
        result = articles_per_publisher(sample_df, top_n=2)
        assert len(result) == 2

    def test_sorted_descending(self, sample_df):
        result = articles_per_publisher(sample_df)
        assert result.iloc[0] >= result.iloc[-1]


class TestPublicationDateTrends:
    def test_returns_series(self, sample_df):
        result = publication_date_trends(sample_df)
        assert isinstance(result, pd.Series)

    def test_weekly_aggregation(self, sample_df):
        result = publication_date_trends(sample_df, freq="W")
        assert result.sum() == len(sample_df)

    def test_monthly_aggregation(self, sample_df):
        result = publication_date_trends(sample_df, freq="M")
        assert result.sum() == len(sample_df)


class TestTopPublishingDays:
    def test_returns_dataframe(self, sample_df):
        result = top_publishing_days(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, sample_df):
        result = top_publishing_days(sample_df)
        assert list(result.columns) == ["date", "article_count"]

    def test_top_n(self, sample_df):
        result = top_publishing_days(sample_df, top_n=2)
        assert len(result) == 2

    def test_sorted_descending(self, sample_df):
        result = top_publishing_days(sample_df)
        assert result["article_count"].iloc[0] >= result["article_count"].iloc[-1]
