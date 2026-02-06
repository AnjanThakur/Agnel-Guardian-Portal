from app.core.database import feedback_forms

feedback_forms.insert_one({
    "form_id": "TEST_FORM_ATLAS",
    "department": "CSE",
    "class": "SE-A"
})

print("MongoDB Atlas connection successful")
