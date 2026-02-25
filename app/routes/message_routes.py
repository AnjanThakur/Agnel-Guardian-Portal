import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from app.core.security import get_current_active_user, users_collection
from app.models.message_schemas import SendMessageRequest, MessageResponse

router = APIRouter(prefix="/messages", tags=["Messages"])

# Configure fastapimail
# In production, these should be robustly pulled from .env
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "dummy@example.com"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", "secret"),
    MAIL_FROM=os.getenv("MAIL_FROM", "noreply@agnel.edu"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

@router.post("/send", response_model=MessageResponse)
async def send_defaulter_message(
    req: SendMessageRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Teacher/Admin endpoint to send a defaulter alert email to a parent.
    """
    if current_user["role"] not in ["teacher", "admin"]:
        raise HTTPException(status_code=403, detail="Only teachers or admins can send alerts.")

    # Find the parent associated with this student_id
    parent = users_collection.find_one({"linked_students": req.student_id, "role": "parent"})
    
    if not parent:
        raise HTTPException(
            status_code=404, 
            detail=f"No parent account found linked to student ID {req.student_id}."
        )

    # Compose the email
    parent_email = parent["email"]
    
    html_body = f"""
    <html>
        <body>
            <p>Dear Parent/Guardian,</p>
            <p>This is an automated academic alert regarding your ward (Student ID: <strong>{req.student_id}</strong>).</p>
            <hr>
            <p><strong>Message from Teacher:</strong></p>
            <p>{req.body}</p>
            <hr>
            <p><small>Regards,<br>Agnel Guardian Portal Administration</small></p>
        </body>
    </html>
    """
    
    message = MessageSchema(
        subject=req.subject,
        recipients=[parent_email],
        body=html_body,
        subtype=MessageType.html
    )

    # Send the email via SMTP in the background
    fm = FastMail(conf)
    try:
        background_tasks.add_task(fm.send_message, message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue email: {str(e)}")

    return MessageResponse(
        status="success", 
        detail=f"Alert email queued for {parent_email}"
    )
