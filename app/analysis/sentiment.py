import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    _sia = SentimentIntensityAnalyzer()


def analyze_sentiment(comments: list[str]):
    summary = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }

    for c in comments:
        score = _sia.polarity_scores(c)["compound"]
        if score >= 0.05:
            summary["positive"] += 1
        elif score <= -0.05:
            summary["negative"] += 1
        else:
            summary["neutral"] += 1

    total = max(1, sum(summary.values()))
    return {
        "counts": summary,
        "percentages": {
            k: round(v * 100 / total, 1)
            for k, v in summary.items()
        }
    }
