from pydantic import BaseModel, EmailStr

class SendMessageRequest(BaseModel):
    # The teacher will select this from a dropdown in the UI
    student_id: str
    subject: str
    body: str

class MessageResponse(BaseModel):
    status: str
    detail: str
