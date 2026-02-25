import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from app.core.security import get_current_active_user, users_collection
from app.core.database import db
from app.models.message_schemas import SendMessageRequest, MessageResponse

router = APIRouter(prefix="/messages", tags=["Messages"])
settings_collection = db["settings"]

# Configure fastapimail
# Connection details are now dynamically generated per-request to support authentic user mailboxes.

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

    # The user explicitly requested to receive the test emails to their own login email.
    # We ignore the parent's email and route the email locally to the sender for verification.
    recipient_email = current_user["email"]
    
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
        recipients=[recipient_email],
        body=html_body,
        subtype=MessageType.html
    )

    # Configure SMTP connection dynamically from the global Admin settings
    smtp_config = settings_collection.find_one({"_id": "smtp_config"})
    if not smtp_config or not smtp_config.get("sender_email") or not smtp_config.get("app_password"):
        raise HTTPException(
            status_code=500, 
            detail="The Mail System has not been configured by an Administrator. Please set the SMTP credentials in the Admin Dashboard."
        )
    
    dynamic_conf = ConnectionConfig(
        MAIL_USERNAME=smtp_config["sender_email"],
        MAIL_PASSWORD=smtp_config["app_password"],
        MAIL_FROM=smtp_config["sender_email"],
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True
    )

    # Send the email via SMTP in the background
    fm = FastMail(dynamic_conf)
    try:
        background_tasks.add_task(fm.send_message, message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue email: {str(e)}")

    return MessageResponse(
        status="success", 
        detail=f"Alert email successfully dispatched regarding {req.student_id}."
    )
