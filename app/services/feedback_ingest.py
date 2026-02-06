from app.core.database import feedback_forms, feedback_ratings
from datetime import datetime

CRITERIA = [
    "The Teaching-Learning Environment",
    "System of Monitoring Students Progress",
    "Involvement of Faculty in Teaching-Learning",
    "Infrastructure Facilities",
    "Learning Resources Like Library, Internet, Computing etc.",
    "Study Environment and Discipline",
    "Counselling and Placements",
    "Institute’s overall Support Facilities",
    "Parental Perception about Institute",
    "Students overall Holistic Development",
]

def save_feedback_form(
    form_id: str,
    department: str,
    class_name: str,
    rating_results: list,
    comment_text: str | None = None
):
    """
    Saves ONE PTA feedback form into MongoDB.
    """

    # Normalize Department Name
    norm_dept = department.strip()
    if norm_dept.lower() in ["computer", "comp", "cs", "general", "cs/it"]:
        norm_dept = "CSE"
    
    # Default Class Name if missing
    if not class_name or not class_name.strip() or class_name == "Unknown":
        class_name = "SE-B"

    # 1️⃣ Save form metadata
    feedback_forms.insert_one({
        "form_id": form_id,
        "department": norm_dept,
        "class": class_name,
        "comment": comment_text or "",
        "created_at": datetime.utcnow()
    })

    # 2️⃣ Save ratings (one document per question)
    for idx, result in enumerate(rating_results):
        if result["status"] != "ok":
            continue  # skip empty / noise / ambiguous for now

        feedback_ratings.insert_one({
            "form_id": form_id,
            "criterion_no": idx + 1,
            "criterion_name": CRITERIA[idx],
            "rating": result["value"],
            "confidence": result["confidence"],
            "department": norm_dept,
            "class": class_name,
        })
