import os
import io
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import PyPDF2
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load Environment Variables
groq_key = os.getenv("GROQ_API_KEY")
pincone_key = os.getenv("PINECONE_API_KEY")

if not groq_key and not pincone_key:
    raise RuntimeError("Missing API keys in .env file.")

#Initialize Clients & Models
client = Groq(api_key=groq_key)
pc=Pinecone(api_key=pincone_key)

# Connect to the Pinecone index
index_name = "research-engine"
pinecone_index = pc.Index(index_name)

# Load the local embedding model (This will download ~90MB the very first time when we run the server)
print("Loading emmbedding model... This might take a few seconds.")
emmbedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app  = FastAPI(title="Research AI Backend", version="3.0")

# Helper function (text Chunking)
def chunk_text(text: str, chunk_size: int = 500):
    """Splits a massive document into smaller chunks of words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@app.get("/")
def health_check():
    return {"status":"operational" , "message" : "RAG Engine is live."}

@app.post("/ask-pdf")
async def ask_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
    ):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # Read and Extract Text
        file_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
        
        # Chunk the text
        text_chunks = chunk_text(extracted_text)

        # Generate Embeddings and Upsert to Pinecone
        # We generate a unique namespace for this specific PDF so data doesn't mix
        doc_namespace = str(uuid.uuid4())

        vectors_to_upsert = []
        for i, chunk in enumerate(text_chunks):
            # Convert text to a 384-dimensional vector
            vector = emmbedding_model.encode(chunk).tolist()

            # Prepare the payload for Pinecone (ID, Vector coordinates, and the raw text Metadata)
            vectors_to_upsert.append({
                "id": f"chunk-{i}",
                "values": vector,
                "metadata": {"text": chunk}
            })

        # Push the data to the cloud database
        pinecone_index.upsert(vectors=vectors_to_upsert, namespace=doc_namespace)

        # Embed the User's Question and Search Pinecone
        question_vector = emmbedding_model.encode(question).tolist()

        # Ask Pinecone for the top 3 most relevant chunks to this specific question
        serach_results = pinecone_index.query(
            vector=question_vector,
            top_k=3,
            namespace=doc_namespace,
            include_metadata=True
        )

        retrieved_context = ""
        for match in serach_results['matches']:
            retrieved_context += match['metadata']['text'] +"\n\n"

        # Generate the Final Answer using Groq
        system_prompt = (
            "You are an elite AI research assistant. Answer the user's question strictly "
            "based on the provided context extracted from a document. If the answer is not in the context, "
            "say 'I cannot find the answer in the provided document.' Do not make up information."
        )

        user_prompt = f"context:\n{retrieved_context}\n\nQuestion: {question}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":user_prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.1
        )

        # Clean up (Optional but good practice: delete the vectors after answering)
        pinecone_index.delete(delete_all=True, namespace=doc_namespace)

        answer = chat_completion.choices[0].message.content 

        return {
        "success": True,
        "filename": file.filename,
        "question": question,
        "answer": answer,
        "analyzed_chunks": len(text_chunks), 
        "model_used": "openai/gpt-oss-120b (Groq)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Pipeline failed: {str(e)}")