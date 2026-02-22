import csv
import io
from app.core.database import students, student_marks

def process_marks_csv(file_content: bytes) -> dict:
    """
    Parses a CSV uploaded by a teacher and stores it in MongoDB.
    Expected headers: student_id, student_name, class_name, department, semester, subject, marks_obtained, max_marks
    """
    decoded = file_content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(decoded))
    
    records_processed = 0
    students_updated = set()

    for row in reader:
        # 1. Extract and clean row data
        sid = row.get("student_id", "").strip()
        sname = row.get("student_name", "").strip()
        sclass = row.get("class_name", "").strip()
        sdept = row.get("department", "").strip()
        
        try:
            sem = int(row.get("semester", 0))
            mobtained = float(row.get("marks_obtained", 0.0))
            mmax = float(row.get("max_marks", 100.0))
        except ValueError:
            continue # skip invalid rows
            
        subject = row.get("subject", "").strip()

        if not sid or not subject or sem < 1:
            continue
            
        # 2. Upsert Student Profile
        if sid not in students_updated:
            students.update_one(
                {"student_id": sid},
                {"$set": {
                    "student_name": sname,
                    "class_name": sclass,
                    "department": sdept
                }},
                upsert=True
            )
            students_updated.add(sid)

        # 3. Upsert Subject Mark for that Semester
        # We store marks individually, or embedded? Let's use a document per subject per semester per student for easy querying.
        student_marks.update_one(
            {
                "student_id": sid,
                "semester": sem,
                "subject": subject
            },
            {"$set": {
                "marks_obtained": mobtained,
                "max_marks": mmax
            }},
            upsert=True
        )
        records_processed += 1
        
    return {
        "status": "success",
        "records_processed": records_processed,
        "unique_students": len(students_updated)
    }
