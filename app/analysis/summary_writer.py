# app/analysis/summary_writer.py
from __future__ import annotations
from typing import Dict, Any, List
import re

# Simple PTA theme map (offline, no LLM)
PTA_THEMES = {
    "Teaching & Faculty": ["teaching", "faculty", "staff", "supportive", "mentor", "explain", "academic"],
    "Practical Exposure": ["practical", "hands-on", "exposure", "lab", "projects", "industry", "application"],
    "Learning Environment": ["environment", "discipline", "focus", "attendance", "classroom", "study"],
    "Infrastructure & Resources": ["infrastructure", "library", "internet", "computing", "facilities", "lab equipment"],
    "Counselling & Placements": ["counselling", "placement", "career", "internship", "guidance"],
    "Holistic Development": ["holistic", "development", "confidence", "skills", "growth", "improvement"],
}

def _theme_hits(keywords: List[str]) -> List[str]:
    hits = {k: 0 for k in PTA_THEMES.keys()}
    for kw in keywords or []:
        low = kw.lower()
        for theme, tokens in PTA_THEMES.items():
            if any(t in low for t in tokens):
                hits[theme] += 1
    ranked = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    return [t for t, c in ranked if c > 0]

def _find_action_words(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in ["need", "improve", "more", "should", "increase", "better", "focus on", "require"])

def generate_paragraph_summary(summary: Dict[str, Any]) -> str:
    """
    Produces a more PTA-meaningful paragraph without paid LLM.
    Uses:
      - sentiment distribution
      - theme mapping from keywords
      - light “action word” cues from topic strings (if available)
    """
    sentiment = summary.get("sentiment") or {}
    keywords = summary.get("keywords") or []
    topics = summary.get("topics") or []

    pos = float(sentiment.get("positive", 0))
    neu = float(sentiment.get("neutral", 0))
    neg = float(sentiment.get("negative", 0))

    # Pick top themes
    themes = _theme_hits(keywords)
    top_themes = themes[:3] if themes else []

    # Detect if there are “improvement” signals from topics/keywords
    combined_text = " ".join(keywords + [str(t) for t in topics])
    has_actions = _find_action_words(combined_text)

    # Sentiment line
    if pos >= max(neu, neg):
        sentiment_line = "Overall parent feedback is predominantly positive."
    elif neg > pos and neg >= neu:
        sentiment_line = "Overall parent feedback highlights several concerns that require attention."
    else:
        sentiment_line = "Parent feedback is mixed, with both strengths and improvement areas noted."

    # Theme line
    if top_themes:
        theme_line = "Key themes most frequently mentioned relate to " + ", ".join(top_themes[:-1]) + (" and " + top_themes[-1] + "." if len(top_themes) > 1 else top_themes[0] + ".")
    else:
        theme_line = "Key themes could not be reliably identified due to limited feedback content."

    # Action line
    if has_actions:
        action_line = "Parents also suggested targeted improvements, indicating opportunities to further strengthen the overall student experience."
    else:
        action_line = "The feedback largely reinforces current strengths, with limited direct requests for change."

    # Close
    close = "These insights can guide focused academic and operational refinements for improved parent satisfaction."

    return " ".join([sentiment_line, theme_line, action_line, close]).strip()
