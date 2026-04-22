from fastapi import APIRouter
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Ollama configuration fallback
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

class SummarizeRequest(BaseModel):
    comments: list[str]

@router.post("/summarize_ai")
async def summarize_comments(request: SummarizeRequest):
    """Summarize extracted comments using Gemini (primary) or local Ollama (fallback)."""
    if not request.comments:
        return {"error": "No comments provided"}

    combined = "\n---\n".join(request.comments)
    prompt = f"""Analyze these parent-teacher feedback comments:

{combined}

Return a JSON object with these exact keys:
- "executive_summary": 2-3 sentence overview of the overall sentiment
- "key_themes": array of objects, each with "theme" (string) and "sentiment" (string: "positive", "negative", or "neutral")
- "actionable_insights": array of 3-5 specific recommendations as strings
- "sentiment_overview": one word - either "Positive", "Negative", or "Mixed"

Return ONLY valid JSON, no markdown formatting or code blocks."""

    # 1. Try Gemini first (Faster)
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash") # Use fast model
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            return json.loads(response.text)
        except Exception as gemini_err:
            print(f"[Gemini Error] Falling back to Ollama: {gemini_err}")

    # 2. Fallback to local Ollama
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120
        )
        response.raise_for_status()
        
        ollama_result = response.json()
        text = ollama_result.get("response", "").strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        
        return json.loads(text)
        
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse AI response", "details": str(e)}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to AI service.", "details": "Gemini failed and Ollama connection refused"}
    except Exception as e:
        return {"error": str(e), "details": "AI AI Summarization failed"}
