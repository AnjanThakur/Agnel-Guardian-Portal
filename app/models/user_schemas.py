from pydantic import BaseModel, EmailStr
from typing import List, Optional
from enum import Enum

class RoleEnum(str, Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"
    parent = "parent"

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: RoleEnum
    # For parents, this is the student(s) they are linking. For students, it's themselves. For admins/teachers, it's empty.
    linked_students: List[str] = []

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    linked_students: Optional[List[str]] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    email: EmailStr
    role: RoleEnum
    linked_students: List[str]
    is_approved: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    
class LinkStudentRequest(BaseModel):
    student_id: str

class AdminUpdateStudentsRequest(BaseModel):
    linked_students: List[str]

class AdminSMTPSettings(BaseModel):
    sender_email: str
    app_password: str
