import os
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.core.security import (
    users_collection, 
    verify_password, 
    get_password_hash, 
    create_access_token, 
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.core.database import db
from app.models.user_schemas import UserCreate, UserUpdate, UserResponse, Token, LinkStudentRequest, AdminUpdateStudentsRequest, AdminSMTPSettings
from app.core.security import require_role
from app.core.database import db

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings_collection = db["settings"]
settings_collection = db["settings"]

@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate):
    # Check if user already exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_password = get_password_hash(user.password)
    # Automatically approve admins/teachers, but require approval for parents/students
    is_approved = True if user.role in ["admin", "teacher"] else False
    
    user_dict = {
        "email": user.email,
        "hashed_password": hashed_password,
        "role": user.role,
        "linked_students": user.linked_students,
        "is_approved": is_approved
    }
    
    users_collection.insert_one(user_dict)
    
    return UserResponse(
        email=user.email,
        role=user.role,
        linked_students=user.linked_students,
        is_approved=is_approved
    )

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"email": form_data.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.get("is_approved", False):
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending admin approval",
        )
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_active_user)):
    return UserResponse(
        email=current_user["email"],
        role=current_user["role"],
        linked_students=current_user.get("linked_students", []),
        is_approved=current_user.get("is_approved", False)
    )

@router.put("/profile", response_model=UserResponse)
def update_profile(updates: UserUpdate, current_user: dict = Depends(get_current_active_user)):
    """
    CRUD Endpoint: Allows a user to update their email, password, or direct links.
    """
    update_data = {}
    
    if updates.email:
        # Verify email uniqueness if changing
        if updates.email != current_user["email"] and users_collection.find_one({"email": updates.email}):
             raise HTTPException(status_code=400, detail="Email already taken.")
        update_data["email"] = updates.email
        
    if updates.password:
        update_data["hashed_password"] = get_password_hash(updates.password)
        
    if updates.linked_students is not None:
        if current_user["role"] != "parent":
             raise HTTPException(status_code=403, detail="Only parents can update linked students arrays.")
        update_data["linked_students"] = updates.linked_students
        
    if update_data:
        users_collection.update_one({"email": current_user["email"]}, {"$set": update_data})
        
    # Re-fetch for response
    updated_user = users_collection.find_one({"_id": current_user["_id"]})
    return UserResponse(
        email=updated_user["email"],
        role=updated_user["role"],
        linked_students=updated_user.get("linked_students", []),
        is_approved=updated_user.get("is_approved", False)
    )

@router.post("/admin/approve/{user_email}", dependencies=[Depends(require_role(["admin", "teacher"]))])
def approve_user(user_email: str):
    """
    Endpoint for Admins/Teachers to approve newly registered Parents/Students.
    """
    user = users_collection.find_one({"email": user_email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
        
    users_collection.update_one({"email": user_email}, {"$set": {"is_approved": True}})
    return {"status": "success", "detail": f"User {user_email} successfully approved."}

@router.post("/link_student", response_model=UserResponse)
def link_student(req: LinkStudentRequest, current_user: dict = Depends(get_current_active_user)):
    """ Endpoint for Parents to manually link a student ID to their account """
    if current_user["role"] != "parent":
        raise HTTPException(status_code=403, detail="Only parents can link students.")
        
    student_id = req.student_id.strip()
    
    current_links = current_user.get("linked_students", [])
    if student_id not in current_links:
        current_links.append(student_id)
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {"linked_students": current_links}}
        )
        
    return UserResponse(
        email=current_user["email"],
        role=current_user["role"],
        linked_students=current_links,
        is_approved=current_user.get("is_approved", False)
    )

@router.get("/admin/users", response_model=list[UserResponse], dependencies=[Depends(require_role(["admin"]))])
def get_all_users():
    """
    Endpoint for Admins to fetch all users in the system to manage them.
    """
    users_cursor = users_collection.find({})
    user_list = []
    
    for doc in users_cursor:
        user_list.append(UserResponse(
            email=doc["email"],
            role=doc["role"],
            linked_students=doc.get("linked_students", []),
            is_approved=doc.get("is_approved", False)
        ))
        
    return user_list

@router.put("/admin/users/{user_email}/students", response_model=UserResponse, dependencies=[Depends(require_role(["admin"]))])
def admin_update_linked_students(user_email: str, req: AdminUpdateStudentsRequest):
    """
    Endpoint for Admins to explicitly overwrite the linked_students array for a given user.
    """
    user = users_collection.find_one({"email": user_email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
        
    if user["role"] not in ["parent", "student"]:
        raise HTTPException(status_code=400, detail="Can only link students to parent or student accounts.")
        
    # Clean up input list (uppercase, strip)
    clean_ids = [sid.strip().upper() for sid in req.linked_students if sid.strip()]
    
    users_collection.update_one(
        {"email": user_email},
        {"$set": {"linked_students": clean_ids}}
    )
    
    return UserResponse(
        email=user["email"],
        role=user["role"],
        linked_students=clean_ids,
        is_approved=user.get("is_approved", False)
    )

@router.get("/admin/settings/smtp", response_model=AdminSMTPSettings, dependencies=[Depends(require_role(["admin"]))])
def get_smtp_settings():
    """
    Fetch the global SMTP settings for the application.
    """
    config = settings_collection.find_one({"_id": "smtp_config"})
    if not config:
        return AdminSMTPSettings(sender_email="", app_password="")
    return AdminSMTPSettings(
        sender_email=config.get("sender_email", ""),
        app_password=config.get("app_password", "")
    )

@router.put("/admin/settings/smtp", response_model=AdminSMTPSettings, dependencies=[Depends(require_role(["admin"]))])
def update_smtp_settings(settings: AdminSMTPSettings):
    """
    Update the global SMTP credentials used to dispatch Teacher emails.
    """
    settings_collection.update_one(
        {"_id": "smtp_config"},
        {"$set": {
            "sender_email": settings.sender_email,
            "app_password": settings.app_password
        }},
        upsert=True
    )
    return settings
