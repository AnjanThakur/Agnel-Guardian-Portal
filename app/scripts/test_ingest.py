from app.services.feedback_ingest import save_feedback_form

# Fake OCR output (simulate pta_inference.py result)
fake_ratings = [
    {"value": 3, "confidence": 0.92, "status": "ok"},
    {"value": 4, "confidence": 0.90, "status": "ok"},
    {"value": 3, "confidence": 0.88, "status": "ok"},
    {"value": 2, "confidence": 0.85, "status": "ok"},
    {"value": 3, "confidence": 0.91, "status": "ok"},
    {"value": 4, "confidence": 0.93, "status": "ok"},
    {"value": 3, "confidence": 0.89, "status": "ok"},
    {"value": 4, "confidence": 0.94, "status": "ok"},
    {"value": 3, "confidence": 0.90, "status": "ok"},
    {"value": 4, "confidence": 0.95, "status": "ok"},
]

save_feedback_form(
    form_id="FORM_TEST_001",
    department="CSE",
    class_name="SE-A",
    rating_results=fake_ratings,
    comment_text="Good teaching environment, infrastructure can improve."
)

print("Feedback form stored successfully")
