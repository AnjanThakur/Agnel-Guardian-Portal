# app/analysis/keywords.py
from __future__ import annotations
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(comments: List[str], top_k: int = 10) -> List[str]:
    if not comments:
        return []

    # filter very short/junk again (safety)
    safe = [c for c in comments if c and len(c.strip()) >= 10]
    if not safe:
        return []

    try:
        # small sets: avoid pruning out everything
        if len(safe) < 4:
            min_df, max_df = 1, 1.0
        else:
            min_df, max_df = 2, 0.85

        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=max_df,
        )
        X = vec.fit_transform(safe)
        if X.shape[1] == 0:
            return []

        scores = X.sum(axis=0).A1
        terms = vec.get_feature_names_out()

        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_k]]

    except Exception:
        return []
