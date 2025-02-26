import os
import re
import csv
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import docx2txt
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from db_manager_sqlite import SQLiteDatabaseManager

class DataProcessor:
    def __init__(self, db_manager: SQLiteDatabaseManager, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the data processor.
        
        Args:
            db_manager: Database manager instance
            embedding_model: Name of the sentence transformer model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.db_manager = db_manager
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def process_file(self, file_path: str) -> str:
        """Process a file and store its chunks in the database.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document ID
        """
        # Extract text from file
        text, metadata = self._extract_text(file_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Store document metadata
        filename = os.path.basename(file_path)
        document_id = self.db_manager.store_document(filename, metadata)
        
        # Process and store each chunk
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.get_embedding(chunk)
            
            # Store chunk with its embedding
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            self.db_manager.store_chunk(document_id, chunk, embedding.tolist(), chunk_metadata)
        
        return document_id
    
    def _extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (extracted text, metadata)
        """
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1].lower()
        
        metadata = {
            "filename": filename,
            "file_type": file_extension[1:] if file_extension else "unknown"
        }
        
        if file_extension == ".pdf":
            return self._extract_from_pdf(file_path, metadata)
        elif file_extension == ".docx":
            return self._extract_from_docx(file_path, metadata)
        elif file_extension == ".txt":
            return self._extract_from_txt(file_path, metadata)
        elif file_extension == ".csv":
            return self._extract_from_csv(file_path, metadata)
        elif file_extension in [".xlsx", ".xls"]:
            return self._extract_from_excel(file_path, metadata)
        elif file_extension == ".json":
            return self._extract_from_json(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_from_pdf(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        text = ""
        with open(file_path, "rb") as file:
            pdf = pypdf.PdfReader(file)
            metadata["page_count"] = len(pdf.pages)
            
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        
        return text, metadata
    
    def _extract_from_docx(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        text = docx2txt.process(file_path)
        return text, metadata
    
    def _extract_from_txt(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        return text, metadata
    
    def _extract_from_csv(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Update metadata
        metadata["row_count"] = len(df)
        metadata["column_count"] = len(df.columns)
        metadata["columns"] = df.columns.tolist()
        
        # Convert DataFrame to text
        text = self._dataframe_to_text(df)
        
        return text, metadata
    
    def _extract_from_excel(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # Update metadata
        metadata["sheet_count"] = len(sheet_names)
        metadata["sheet_names"] = sheet_names
        
        # Process each sheet
        all_text = []
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += self._dataframe_to_text(df)
            all_text.append(sheet_text)
        
        return "\n\n".join(all_text), metadata
    
    def _extract_from_json(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Extract text from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            metadata: Existing metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Convert JSON to text
        if isinstance(data, list):
            metadata["item_count"] = len(data)
            if len(data) > 0 and isinstance(data[0], dict):
                # Try to convert to DataFrame if it's a list of objects
                df = pd.DataFrame(data)
                text = self._dataframe_to_text(df)
            else:
                # Otherwise, just pretty print
                text = json.dumps(data, indent=2)
        else:
            # For non-list JSON, pretty print
            text = json.dumps(data, indent=2)
        
        return text, metadata
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert a DataFrame to text format.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Text representation of the DataFrame
        """
        # Get column names
        columns = df.columns.tolist()
        
        # Build text representation
        text = "Columns: " + ", ".join(columns) + "\n\n"
        
        # Add rows
        for i, row in df.iterrows():
            text += f"Row {i}:\n"
            for col in columns:
                text += f"  {col}: {row[col]}\n"
            text += "\n"
        
        return text
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text)
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            
        Returns:
            List of similar chunks with their documents
        """
        # Generate embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Search for similar chunks
        return self.db_manager.search_similar_chunks(query_embedding.tolist(), limit)
