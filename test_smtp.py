import asyncio
from app.core.database import db
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

settings_collection = db["settings"]

async def test_email():
    print("Fetching SMTP Settings from Database...")
    smtp_config = settings_collection.find_one({"_id": "smtp_config"})
    if not smtp_config:
        print("ERROR: smtp_config not found in database.")
        return
        
    sender = smtp_config.get("sender_email")
    pwd = smtp_config.get("app_password")
    
    if not sender or not pwd:
        print("ERROR: Sender email or password missing from DB.")
        return
        
    print(f"Loaded Sender: {sender}")
    
    dynamic_conf = ConnectionConfig(
        MAIL_USERNAME=sender,
        MAIL_PASSWORD=pwd,
        MAIL_FROM=sender,
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True
    )
    
    # Send to the same address for testing
    message = MessageSchema(
        subject="Sync SMTP Test",
        recipients=[sender],
        body="If you receive this, SMTP is working perfectly!",
        subtype=MessageType.html
    )

    fm = FastMail(dynamic_conf)
    print("Initiating synchronous sending request to smtp.gmail.com...")
    try:
        await fm.send_message(message)
        print("SUCCESS! The email was accepted by Google and dispatched.")
    except Exception as e:
        print(f"FAILED TO SEND! Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_email())
