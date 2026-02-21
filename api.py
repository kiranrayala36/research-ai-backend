import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("API key not found. Please set GROQ_API_KEY in your environment variables.")

client = Groq(api_key=api_key)

app  = FastAPI(title="Research AI Backend", version="2.0")

class DocumentRequest(BaseModel):
    title: str 
    content: str 
    max_summary_length: int = 150

@app.get("/")
def health_check():
    return {"status":"operational" , "message" : "Groq AI Backend is live."}

@app.post("/process-document")
def process_document(doc: DocumentRequest):
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")
    
    system_prompt = (
        "You are an elite AI research assistant. Your job is to read the provided text "
        "and extract the core scientific or technical insights. "
        f"Keep your response concise, professional, and strictly under {doc.max_summary_length} words."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":f"Title: {doc.title}\n\nContent: {doc.content}"}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.3
        )

        ai_insights = chat_completion.choices[0].message.content 
        return {
        "success": True,
        "document_tile": doc.title,
        "insights": ai_insights,
        "model_used": "openai/gpt-oss-120b (Groq)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Inference failed: {str(e)}")