from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
import uuid

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup ChromaDB
CHROMA_DIR = "chroma_db"
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))#, chroma_db_impl="duckdb+parquet"))
# client = chromadb.Client(persist_directory="chroma_db")
collection = client.get_or_create_collection("resumes")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

UPLOAD_DIR = "resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile, candidate_name: str = Form(...)):
    # Save PDF
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract & embed
    text = extract_text_from_pdf(file_path)
    embedding = model.encode(text).tolist()

    # Store in ChromaDB
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[file_id],
        metadatas=[{"name": candidate_name, "filename": file.filename}]
    )

    return {"status": "success", "id": file_id}

@app.post("/rank/")
async def rank_resumes(job_description: str, top_k: int = 5):
    query_embedding = model.encode(job_description).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # Return ranked resumes
    ranked = []
    for i in range(len(results["ids"][0])):
        ranked.append({
            "id": results["ids"][0][i],
            "candidate_name": results["metadatas"][0][i]["name"],
            "filename": results["metadatas"][0][i]["filename"],
            "score": results["distances"][0][i]
        })

    return JSONResponse(content=ranked)