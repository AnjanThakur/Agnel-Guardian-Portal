from fastapi import APIRouter
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class SummarizeRequest(BaseModel):
    comments: list[str]

@router.post("/summarize_ai")
async def summarize_comments(request: SummarizeRequest):
    """Summarize extracted comments using Gemini API."""
    if not request.comments:
        return {"error": "No comments provided"}

    combined = "\n---\n".join(request.comments)
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Analyze these parent-teacher feedback comments:

{combined}

Return a JSON object with these exact keys:
- "executive_summary": 2-3 sentence overview of the overall sentiment
- "key_themes": array of objects, each with "theme" (string) and "sentiment" (string: "positive", "negative", or "neutral")
- "actionable_insights": array of 3-5 specific recommendations as strings
- "sentiment_overview": one word - either "Positive", "Negative", or "Mixed"

Return ONLY valid JSON, no markdown formatting or code blocks."""

        response = model.generate_content(prompt)
        
        # Extract text and parse JSON
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        result = json.loads(text)
        return result
        
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse AI response", "details": str(e), "raw": response.text if 'response' in dir() else "No response"}
    except Exception as e:
        return {"error": str(e), "details": "Gemini API call failed"}
