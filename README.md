# Resume Tracker  
A FastAPI-based application to upload resumes, store embeddings, and rank candidates against job descriptions using semantic search.  

## Features  
- Upload candidate resumes (PDF).  
- Automatically extract text from resumes.  
- Generate embeddings using `SentenceTransformer (all-MiniLM-L6-v2)`.  
- Store embeddings in ChromaDB for fast retrieval.  
- Rank resumes against a job description query.  
- Simple HTML frontend for uploading and ranking.

## Tech Stack  
- **FastAPI** – backend framework    
- **PyMuPDF (fitz)** – PDF text extraction  
- **SentenceTransformers** – embeddings  
- **ChromaDB** – vector database for similarity search  
- **Uvicorn** – ASGI server  
- **Jinja2** – templating (frontend)

## Installation  
  
1. Clone the repo   

`git clone https://github.com/yourusername/resume_tracker.git  
cd resume_tracker `  
    
`pip install -r requirements.txt`  
 2. open in browser `http://127.0.0.1:8000/`    
  
## Project Structure  
resume_tracker/  
 ├── main.py              # FastAPI backend  
 ├── templates/
 │   └── index.html       # Frontend HTML  
 ├── resumes/             # Uploaded resumes  
 ├── chroma_db/           # Vector DB storage  
 ├── requirements.txt  
 └── README.md  
