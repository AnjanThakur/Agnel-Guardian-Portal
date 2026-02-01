# app/analysis/run_full_analysis.py
import os
from typing import List, Dict, Any

from app.analysis.sanatize_text import sanitize_for_analysis
from app.analysis.feedback_summary import summarize_feedback
from app.analysis.summary_writer import generate_paragraph_summary
from app.analysis.visualize import plot_sentiment, plot_keywords, plot_topics
from app.analysis.pdf_report import export_feedback_report_pdf

def run_full_feedback_analysis(comments, report_dir):
    print("ANALYSIS INPUT:", repr(comments))
    print("TEXT LENGTH:", len(comments[0]) if comments and comments[0] else 0)

def _is_junk(s: str) -> bool:
    if not s:
        return True
    t = s.strip()
    if len(t) < 10:
        return True
    low = t.lower()
    # repeated chars or alpha-only noise
    if low.isalpha() and len(set(low)) <= 4:
        return True
    return False

def run_full_feedback_analysis(
    comments: List[str],
    out_dir: str,
    institute_name: str = "Institute",
) -> Dict[str, Any]:

    os.makedirs(out_dir, exist_ok=True)

    # sanitize + filter
    clean_comments: List[str] = []
    for c in comments or []:
        if _is_junk(c):
            continue
        sc = sanitize_for_analysis(c)
        if not _is_junk(sc):
            clean_comments.append(sc)

    summary = summarize_feedback(clean_comments)
    paragraph = generate_paragraph_summary(summary)

    # charts
    sentiment_path = os.path.join(out_dir, "sentiment.png")
    keywords_path = os.path.join(out_dir, "keywords.png")
    topics_path = os.path.join(out_dir, "topics.png")

    plot_sentiment(summary.get("sentiment", {}), sentiment_path)
    plot_keywords(summary.get("keywords", []), keywords_path)
    plot_topics(summary.get("topics", []), topics_path)

    # pdf
    pdf_path = os.path.join(out_dir, "PTA_Feedback_Report.pdf")
    export_feedback_report_pdf(
        output_path=pdf_path,
        institute_name=institute_name,
        summary=summary,
        paragraph_summary=paragraph,
        charts={
            "Sentiment Distribution": sentiment_path,
            "Top Keywords": keywords_path,
            "Topic Overview": topics_path,
        },
    )

    return {
        "summary": summary,
        "paragraph": paragraph,
        "files": {
            "pdf": pdf_path,
            "sentiment_chart": sentiment_path,
            "keywords_chart": keywords_path,
            "topics_chart": topics_path,
        },
        "clean_comments_used": len(clean_comments),
    }
