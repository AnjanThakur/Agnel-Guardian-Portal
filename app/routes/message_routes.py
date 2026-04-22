import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from app.core.security import get_current_active_user, users_collection
from app.core.database import db
from app.models.message_schemas import SendMessageRequest, MessageResponse

router = APIRouter(prefix="/messages", tags=["Messages"])

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

    # Route the email to the linked parent's registered email address
    recipient_email = parent["email"]
    
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

    # Configure SMTP connection dynamically from environment variables
    mail_username = os.getenv("MAIL_USERNAME")
    mail_password = os.getenv("MAIL_PASSWORD")
    
    if not mail_username or not mail_password:
        raise HTTPException(
            status_code=500, 
            detail="Server Misconfiguration: MAIL_USERNAME and MAIL_PASSWORD are not set in the .env file."
        )
    
    dynamic_conf = ConnectionConfig(
        MAIL_USERNAME=mail_username,
        MAIL_PASSWORD=mail_password,
        MAIL_FROM=mail_username,
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=False
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
