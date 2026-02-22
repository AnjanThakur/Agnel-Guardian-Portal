from pydantic import BaseModel
from typing import List, Optional

class MarkEntry(BaseModel):
    subject: str
    marks_obtained: float
    max_marks: float

class SemesterMarks(BaseModel):
    semester: int
    marks: List[MarkEntry]
    sgpa: Optional[float] = None
    percentage: Optional[float] = None

class StudentProfile(BaseModel):
    student_id: str
    student_name: str
    class_name: str
    department: str

class StudentMarksResponse(BaseModel):
    profile: StudentProfile
    semesters: List[SemesterMarks]
