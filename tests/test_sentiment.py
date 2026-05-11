import numpy as np
import pandas as pd
import pytest
from src.sentiment import (
    score_sentiment,
    sentiment_category,
    compute_daily_returns,
    align_to_next_trading_day,
    aggregate_daily_sentiment,
    pearson_per_ticker,
)


@pytest.fixture
def trading_dates():
    # Mon 2021-01-04 through Fri 2021-01-08 (US trading days)
    return pd.DatetimeIndex(pd.date_range("2021-01-04", "2021-01-08", freq="B"))


@pytest.fixture
def sample_stock():
    idx = pd.date_range("2021-01-04", "2021-01-08", freq="B", name="date")
    return pd.DataFrame(
        {"Close": [130.0, 132.0, 131.0, 135.0, 133.0]},
        index=idx,
    )


@pytest.fixture
def sample_news():
    return pd.DataFrame(
        {
            "headline": [
                "Apple surges to record highs on strong earnings",
                "Apple stock drops after disappointing guidance",
                "Apple maintains steady growth",
                "Apple announces major product launch",
            ],
            "stock": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "pub_date": pd.to_datetime(
                ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
            ),
            "sentiment": [0.7, -0.5, 0.0, 0.6],
        }
    )


class TestScoreSentiment:
    def test_positive_headline(self):
        s = score_sentiment(pd.Series(["Excellent earnings beat all expectations!"]))
        assert s.iloc[0] > 0.05

    def test_negative_headline(self):
        s = score_sentiment(pd.Series(["Market crashes amid recession fears"]))
        assert s.iloc[0] < 0.0

    def test_returns_series(self):
        headlines = pd.Series(["Good news", "Bad news", "No news"])
        result = score_sentiment(headlines)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_nan_input(self):
        result = score_sentiment(pd.Series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_score_bounds(self):
        headlines = pd.Series(["Amazing great fantastic!", "Terrible horrible disaster!"])
        result = score_sentiment(headlines)
        assert all(result.between(-1, 1))


class TestSentimentCategory:
    def test_positive(self):
        assert sentiment_category(0.5) == "positive"

    def test_negative(self):
        assert sentiment_category(-0.5) == "negative"

    def test_neutral_positive_boundary(self):
        assert sentiment_category(0.04) == "neutral"

    def test_neutral_negative_boundary(self):
        assert sentiment_category(-0.04) == "neutral"

    def test_exactly_at_positive_threshold(self):
        # threshold is strict (> 0.05), so 0.05 maps to neutral
        assert sentiment_category(0.05) == "neutral"

    def test_exactly_at_negative_threshold(self):
        # threshold is strict (< -0.05), so -0.05 maps to neutral
        assert sentiment_category(-0.05) == "neutral"


class TestComputeDailyReturns:
    def test_first_row_is_nan(self, sample_stock):
        r = compute_daily_returns(sample_stock)
        assert pd.isna(r.iloc[0])

    def test_correct_return(self, sample_stock):
        r = compute_daily_returns(sample_stock)
        expected = (132.0 - 130.0) / 130.0 * 100
        assert abs(r.iloc[1] - expected) < 1e-6

    def test_returns_series(self, sample_stock):
        r = compute_daily_returns(sample_stock)
        assert isinstance(r, pd.Series)
        assert r.name == "daily_return"

    def test_length_matches_input(self, sample_stock):
        r = compute_daily_returns(sample_stock)
        assert len(r) == len(sample_stock)


class TestAlignToNextTradingDay:
    def test_weekday_unchanged(self, trading_dates):
        pub = pd.Series(pd.to_datetime(["2021-01-04"]))  # Monday
        aligned = align_to_next_trading_day(pub, trading_dates)
        assert aligned.iloc[0] == pd.Timestamp("2021-01-04")

    def test_saturday_goes_to_monday(self, trading_dates):
        # 2021-01-02 was a Saturday; next trading day is 2021-01-04 (Monday)
        pub = pd.Series(pd.to_datetime(["2021-01-02"]))
        aligned = align_to_next_trading_day(pub, trading_dates)
        assert aligned.iloc[0] == pd.Timestamp("2021-01-04")

    def test_sunday_goes_to_monday(self, trading_dates):
        pub = pd.Series(pd.to_datetime(["2021-01-03"]))
        aligned = align_to_next_trading_day(pub, trading_dates)
        assert aligned.iloc[0] == pd.Timestamp("2021-01-04")

    def test_nan_input_returns_nat(self, trading_dates):
        pub = pd.Series([pd.NaT])
        aligned = align_to_next_trading_day(pub, trading_dates)
        assert pd.isna(aligned.iloc[0])

    def test_date_beyond_calendar_returns_nat(self, trading_dates):
        pub = pd.Series(pd.to_datetime(["2025-01-01"]))
        aligned = align_to_next_trading_day(pub, trading_dates)
        assert pd.isna(aligned.iloc[0])


class TestAggregateDailySentiment:
    def test_returns_dataframe(self, sample_news, trading_dates):
        result = aggregate_daily_sentiment(sample_news, trading_dates, "AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_news, trading_dates):
        result = aggregate_daily_sentiment(sample_news, trading_dates, "AAPL")
        assert "date" in result.columns
        assert "sentiment_score" in result.columns
        assert "article_count" in result.columns

    def test_article_count_positive(self, sample_news, trading_dates):
        result = aggregate_daily_sentiment(sample_news, trading_dates, "AAPL")
        assert (result["article_count"] > 0).all()

    def test_mean_sentiment_in_range(self, sample_news, trading_dates):
        result = aggregate_daily_sentiment(sample_news, trading_dates, "AAPL")
        assert result["sentiment_score"].between(-1, 1).all()


class TestPearsonPerTicker:
    def test_returns_dataframe(self, sample_news, sample_stock):
        merged = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "sentiment_score": [0.7, -0.5, 0.0, 0.6, 0.3, -0.2],
                "daily_return": [1.5, -2.0, 0.3, 0.8, 1.1, -0.4],
            }
        )
        result = pearson_per_ticker(merged)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        merged = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "sentiment_score": np.linspace(-1, 1, 10),
                "daily_return": np.linspace(-2, 2, 10),
            }
        )
        result = pearson_per_ticker(merged)
        for col in ["ticker", "pearson_r", "p_value", "n_days", "significant"]:
            assert col in result.columns

    def test_perfect_positive_correlation(self):
        x = np.linspace(0, 1, 20)
        merged = pd.DataFrame(
            {"ticker": ["TEST"] * 20, "sentiment_score": x, "daily_return": x * 2}
        )
        result = pearson_per_ticker(merged)
        assert abs(result.iloc[0]["pearson_r"] - 1.0) < 1e-6

    def test_insufficient_data_excluded(self):
        merged = pd.DataFrame(
            {
                "ticker": ["TINY"] * 3,
                "sentiment_score": [0.1, 0.2, 0.3],
                "daily_return": [0.5, 0.6, 0.7],
            }
        )
        result = pearson_per_ticker(merged)
        assert len(result) == 0
