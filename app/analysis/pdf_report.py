from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from datetime import datetime
import os


def export_feedback_report_pdf(
    output_path: str,
    institute_name: str,
    summary: dict,
    paragraph_summary: str,
    charts: dict,
):
    """
    Creates a professional PTA feedback analysis PDF report.
    """

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    # ---------------- Title ----------------
    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER

    story.append(Paragraph(institute_name, title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("PTA Feedback Analysis Report", styles["Heading2"]))
    story.append(Spacer(1, 6))

    date_str = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph(f"<i>Generated on {date_str}</i>", styles["Normal"]))
    story.append(Spacer(1, 20))

    # ---------------- Executive Summary ----------------
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(paragraph_summary, styles["BodyText"]))
    story.append(Spacer(1, 20))

    # ---------------- Key Statistics ----------------
    story.append(Paragraph("Key Statistics", styles["Heading2"]))
    story.append(Spacer(1, 6))

    sentiment = summary["sentiment"]["percentages"]
    total_comments = summary["total_comments"]

    stats_table = Table(
        [
            ["Total Feedback Entries", str(total_comments)],
            ["Positive Feedback (%)", str(sentiment["positive"])],
            ["Neutral Feedback (%)", str(sentiment["neutral"])],
            ["Negative Feedback (%)", str(sentiment["negative"])],
        ],
        colWidths=[220, 120],
    )

    stats_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONT", (0, 0), (-1, -1), "Helvetica"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    story.append(stats_table)
    story.append(Spacer(1, 20))

    # ---------------- Charts ----------------
    story.append(Paragraph("Visual Analysis", styles["Heading2"]))
    story.append(Spacer(1, 10))

    for title, img_path in charts.items():
        if not os.path.exists(img_path):
            continue

        story.append(Paragraph(title.replace("_", " ").title(), styles["Heading3"]))
        story.append(Spacer(1, 6))
        story.append(RLImage(img_path, width=420, height=240))
        story.append(Spacer(1, 20))

    # ---------------- Build ----------------
    doc.build(story)
