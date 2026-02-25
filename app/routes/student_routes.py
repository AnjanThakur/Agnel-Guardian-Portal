from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.core.database import students, student_marks
from app.services.marks_ingest import process_marks_csv
from app.models.marks_schemas import StudentMarksResponse, StudentProfile, SemesterMarks, MarkEntry
from app.core.security import get_current_active_user, require_role

router = APIRouter(prefix="/student", tags=["Student Marks"])

@router.post("/admin/marks/upload", dependencies=[Depends(require_role(["admin", "teacher"]))])
async def upload_marks_csv(file: UploadFile = File(...)):
    """
    Endpoint for Admin/Teachers to bulk upload student marks via CSV.
    Requires Admin/Teacher JWT role.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    content = await file.read()
    result = process_marks_csv(content)
    return result

@router.get("/{student_id}", response_model=StudentMarksResponse)
def get_student_marks_dashboard(
    student_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for Parents to view their child's profile and marks progression across all mapped semesters.
    Protected by role checks.
    """
    role = current_user.get("role")
    
    # Simple ACL check: If they are a parent/student, verify this student_id is within their linked_students
    if role in ["parent", "student"]:
        linked_students = current_user.get("linked_students", [])
        if student_id not in linked_students:
             raise HTTPException(status_code=403, detail="You do not have authorization to view this student's profile.")
             
    # 1. Fetch Profile
    student_doc = students.find_one({"student_id": student_id})
    if not student_doc:
         raise HTTPException(status_code=404, detail="Student not found.")
         
    profile = StudentProfile(
        student_id=student_doc.get("student_id", ""),
        student_name=student_doc.get("student_name", "Unknown"),
        class_name=student_doc.get("class_name", "Unknown"),
        department=student_doc.get("department", "Unknown"),
    )
    
    # 2. Fetch all marks and group by semester
    marks_cursor = student_marks.find({"student_id": student_id})
    sem_dict = {}
    
    for doc in marks_cursor:
        sem = doc["semester"]
        subject = doc["subject"]
        obtained = doc["marks_obtained"]
        max_m = doc["max_marks"]
        
        if sem not in sem_dict:
            sem_dict[sem] = {
                "marks": [],
                "total_obtained": 0.0,
                "total_max": 0.0
            }
            
        sem_dict[sem]["marks"].append(MarkEntry(
            subject=subject,
            marks_obtained=obtained,
            max_marks=max_m
        ))
        sem_dict[sem]["total_obtained"] += obtained
        sem_dict[sem]["total_max"] += max_m
        
    # 3. Calculate aggregates
    semesters = []
    for sem, data in sorted(sem_dict.items()):
        percentage = None
        if data["total_max"] > 0:
            percentage = round((data["total_obtained"] / data["total_max"]) * 100, 2)
            
        # Mock SGPA approximation if percentage is present (typically (percentage/10) + 0.75 or similar, but let's just use percentage / 9.5 for rough mock)
        sgpa = round(percentage / 9.5, 2) if percentage else None
        
        semesters.append(SemesterMarks(
            semester=sem,
            marks=data["marks"],
            percentage=percentage,
            sgpa=sgpa
        ))
        
    return StudentMarksResponse(profile=profile, semesters=semesters)
