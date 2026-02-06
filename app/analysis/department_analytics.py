from app.core.database import feedback_ratings

def get_department_criterion_wise_distribution(department: str, class_name: str | None = None):
    """
    Returns rating distribution (1–4) for EACH criterion in a department
    """

    # Normalize synonyms for CSE
    target_depts = [department]
    if department.upper() == "CSE":
        target_depts = ["CSE", "Computer", "Comp", "CS", "General", "CS/IT"]

    match_query = {
        "department": {"$in": target_depts},
        "rating": {"$ne": None}
    }
    
    if class_name:
        match_query["class"] = class_name

    pipeline = [
        {
            "$match": match_query
        },
        {
            "$group": {
                "_id": {
                    "criterion": "$criterion_name",
                    "rating": "$rating"
                },
                "count": {"$sum": 1}
            }
        }
    ]

    raw = list(feedback_ratings.aggregate(pipeline))

    result = {}

    for r in raw:
        criterion = r["_id"]["criterion"]
        rating = str(r["_id"]["rating"])
        count = r["count"]

        # Initialize criterion if not exists
        if criterion not in result:
            result[criterion] = {"1": 0, "2": 0, "3": 0, "4": 0}

        result[criterion][rating] = count

    return {
        "department": department,
        "criteria": result
    }
