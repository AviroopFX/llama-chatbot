import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from typing import List, Dict, Any, Optional
import json

from db_manager_sqlite import SQLiteDatabaseManager
from data_processor import DataProcessor
from llama_model import LlamaModel

# Initialize FastAPI app
app = FastAPI(title="Custom Llama Chatbot (SQLite Version)")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize database manager
db_manager = SQLiteDatabaseManager()

# Initialize data processor
data_processor = DataProcessor(db_manager)

# Initialize Llama model
MODEL_PATH = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf")  # Default model path
llama_model = None  # Will be initialized when a model file is available

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    # Get list of documents
    documents = db_manager.get_all_documents()
    
    # Check if model is loaded
    model_loaded = llama_model is not None
    available_models = []
    
    # Check for available models
    if os.path.exists("models"):
        available_models = [f for f in os.listdir("models") if f.endswith(".gguf")]
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "documents": documents,
            "model_loaded": model_loaded,
            "available_models": available_models
        }
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file."""
    # Save the file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file
    try:
        document_id = data_processor.process_file(file_path)
        return {"status": "success", "message": f"File processed successfully", "document_id": document_id}
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}"}

@app.post("/load-model")
async def load_model(model_filename: str = Form(...)):
    """Load a Llama model."""
    global llama_model
    
    model_path = os.path.join("models", model_filename)
    
    if not os.path.exists(model_path):
        return {"status": "error", "message": f"Model file not found: {model_filename}"}
    
    try:
        llama_model = LlamaModel(model_path)
        return {"status": "success", "message": f"Model loaded successfully: {model_filename}"}
    except Exception as e:
        return {"status": "error", "message": f"Error loading model: {str(e)}"}

@app.post("/chat")
async def chat(query: str = Form(...), top_k: int = Form(5)):
    """Process a chat query."""
    if llama_model is None:
        return {"status": "error", "message": "No model loaded. Please load a model first."}
    
    # Search for relevant context
    context = data_processor.search_similar(query, limit=top_k)
    
    if not context:
        return {"response": "I don't have any relevant information to answer that question."}
    
    # Generate response
    response = llama_model.generate_response(query, context)
    
    # Format sources for display
    sources = []
    for item in context:
        document = item["document"]
        sources.append({
            "filename": document["filename"],
            "similarity": item["similarity_score"]
        })
    
    return {
        "response": response,
        "sources": sources
    }

@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    success = db_manager.delete_document(document_id)
    
    if success:
        return {"status": "success", "message": "Document deleted successfully"}
    else:
        return {"status": "error", "message": "Error deleting document"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon."""
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else None

if __name__ == "__main__":
    import os
    import argparse
    
    # Create necessary directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    # Use environment port if available (for Replit)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run("app_sqlite:app", host=args.host, port=port, reload=True)
