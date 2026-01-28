from app.analysis.text_preprocess import normalize_text
from app.analysis.keywords import extract_keywords
from app.analysis.topics import extract_topics
from app.analysis.sentiment import analyze_sentiment

def summarize_feedback(comments_raw: list[str]) -> dict:
    cleaned = [c for c in comments_raw if c.strip()]
    normalized = [normalize_text(c) for c in cleaned]

    return {
        "total_comments": len(cleaned),
        "keywords": extract_keywords(normalized),
        "topics": extract_topics(normalized),
        "sentiment": analyze_sentiment(cleaned),
    }
