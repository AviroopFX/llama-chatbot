import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

# Import your custom modules
from db_manager_sqlite import SQLiteDatabaseManager
from data_processor import DataProcessor

# Initialize database and processor
db_path = os.path.join(os.path.dirname(__file__), 'chatbot.db')
db_manager = SQLiteDatabaseManager(db_path)
data_processor = DataProcessor(db_manager)

# Create FastAPI app
app = FastAPI(
    title="AI Chatbot",
    description="A smart chatbot for processing and querying documents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a file
    """
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        
        with open(file_path, 'wb') as buffer:
            buffer.write(await file.read())
        
        # Process the file
        document_id = data_processor.process_file(file_path)
        
        return JSONResponse(content={
            "message": "File uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(query: str):
    """
    Search for similar chunks based on a query
    """
    try:
        # Perform similarity search
        similar_chunks = data_processor.search_similar(query)
        
        return JSONResponse(content={
            "query": query,
            "results": similar_chunks
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
