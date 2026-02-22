import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.database import students, student_marks
from app.models.marks_schemas import StudentMarksResponse, StudentProfile, SemesterMarks, MarkEntry

def test_fetch():
    student_id = "S001"
    print(f"Fetching profile for {student_id}...")
    student_doc = students.find_one({"student_id": student_id})
    print(student_doc)
    
    marks_cursor = student_marks.find({"student_id": student_id})
    print(list(marks_cursor))
    
if __name__ == "__main__":
    test_fetch()
