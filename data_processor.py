import os
import re
import csv
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from db_manager_sqlite import SQLiteDatabaseManager
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, db_manager: SQLiteDatabaseManager, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 min_chunk_length: int = 50):
        """Initialize the data processor with enhanced chunking.
        
        Args:
            db_manager: Database manager instance
            embedding_model: Name of the sentence transformer model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            min_chunk_length: Minimum length of a chunk to be processed
        """
        self.db_manager = db_manager
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]  # More granular splitting
        )
        self.min_chunk_length = min_chunk_length
    
    def process_file(self, file_path: str) -> str:
        """Enhanced file processing with improved chunk handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document ID
        """
        # Clear existing document and chunks for this file
        filename = os.path.basename(file_path)
        existing_doc = self.db_manager.get_document_by_filename(filename)
        if existing_doc:
            self.db_manager.delete_document(existing_doc['id'])
        
        # Extract text from file
        text, metadata = self._extract_text(file_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_length]
        
        # Store document metadata
        document_id = self.db_manager.store_document(filename, metadata)
        
        # Process and store each chunk
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = self.get_embedding(chunk)
            
            # Store chunk with its embedding
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk)
            }
            self.db_manager.store_chunk(document_id, chunk, embedding.tolist(), chunk_metadata)
        
        return document_id
    
    def _extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from various file types.
        
        Args:
            file_path: Path to the file to extract text from
        
        Returns:
            Tuple of extracted text and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        metadata = {
            "filename": os.path.basename(file_path),
            "file_type": file_extension
        }
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path), metadata
            elif file_extension == '.json':
                logger.info(f"Processing JSON file: {file_path}")
                text = self._extract_from_json(file_path)
                logger.info(f"Extracted JSON text:\n{text}")
                return text, metadata
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path), metadata
            elif file_extension == '.csv':
                return self._extract_from_csv(file_path), metadata
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path), metadata
            elif file_extension in [".xlsx", ".xls"]:
                return self._extract_from_excel(file_path), metadata
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return "", metadata
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Enhanced PDF text extraction with improved parsing and cleaning.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Extracted text
        """
        import re
        from pypdf import PdfReader
        
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                metadata["page_count"] = len(pdf.pages)
                
                # Enhanced text extraction
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    
                    # Advanced text cleaning
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                    page_text = re.sub(r'^\s+|\s+$', '', page_text)  # Trim
                    page_text = re.sub(r'[^\x00-\x7F]+', '', page_text)  # Remove non-ASCII
                    
                    # Add structural context
                    page_text = f"--- Page {page_num} ---\n{page_text}\n"
                    text += page_text
                
                # Additional metadata
                metadata["total_text_length"] = len(text)
                metadata["avg_chars_per_page"] = len(text) / metadata["page_count"] if metadata["page_count"] > 0 else 0
                
                # Optional: Extract document title or first heading
                first_page_text = pdf.pages[0].extract_text()
                potential_title = re.search(r'^([A-Z][^\n]+)', first_page_text, re.MULTILINE)
                if potential_title:
                    metadata["document_title"] = potential_title.group(1).strip()
        
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            text = f"Error extracting PDF: {e}"
        
        return text
    
    def _extract_from_json(self, file_path: str) -> str:
        """Extract text from JSON file with improved readability.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            Extracted text as a string
        """
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Flatten JSON to more readable text
        def extract_details(obj):
            details = []
            
            # Handle list of dictionaries (typical JSON structure)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        # Extract key details from each employee/item
                        employee_details = []
                        
                        # Core details
                        if 'employee_id' in item:
                            employee_details.append(f"Employee ID: {item['employee_id']}")
                        if 'name' in item:
                            employee_details.append(f"Name: {item['name']}")
                        if 'department' in item:
                            employee_details.append(f"Department: {item['department']}")
                        if 'position' in item:
                            employee_details.append(f"Position: {item['position']}")
                        
                        # Skills
                        if 'skills' in item and isinstance(item['skills'], list):
                            skills = ', '.join(str(skill) for skill in item['skills'])
                            employee_details.append(f"Skills: {skills}")
                        
                        # Projects
                        if 'projects' in item and isinstance(item['projects'], list):
                            project_details = []
                            for project in item['projects']:
                                project_info = []
                                if 'name' in project:
                                    project_info.append(f"Project: {project['name']}")
                                if 'status' in project:
                                    project_info.append(f"Status: {project['status']}")
                                if project_info:
                                    project_details.append(' - '.join(project_info))
                            
                            if project_details:
                                employee_details.append("Projects: " + '; '.join(project_details))
                        
                        details.append('. '.join(employee_details))
            
            return '\n'.join(details)
        
        # Convert to text
        return extract_details(data)
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
        
        Returns:
            Extracted text
        """
        text = docx2txt.process(file_path)
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
        
        Returns:
            Extracted text
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        return text
    
    def _extract_from_csv(self, file_path: str) -> str:
        """Extract text from a CSV file.
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            Extracted text
        """
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Update metadata
        metadata["row_count"] = len(df)
        metadata["column_count"] = len(df.columns)
        metadata["columns"] = df.columns.tolist()
        
        # Convert DataFrame to text
        text = self._dataframe_to_text(df)
        
        return text
    
    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from an Excel file.
        
        Args:
            file_path: Path to the Excel file
        
        Returns:
            Extracted text
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
        
        return "\n\n".join(all_text)
    
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
        """Generate a standardized 384-dimensional embedding for the given text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Normalized 384-dimensional embedding vector
        """
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return np.zeros(384, dtype=np.float32)  # Default zero vector
        
        # Preprocess text to improve embedding quality
        # Remove extra whitespace and convert to lowercase
        text = ' '.join(text.split()).lower()
        
        # Generate embedding using the specific model
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Ensure 384-dimensional embedding
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # Truncate or pad to exactly 384 dimensions
        if len(embedding) > 384:
            embedding = embedding[:384]
        elif len(embedding) < 384:
            embedding = np.pad(embedding, (0, 384 - len(embedding)), mode='constant')
        
        # Normalize the embedding
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def search_similar(self, query: str, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Advanced context search with improved filtering and ranking.
        
        Args:
            query: Search query string
            limit: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score to include a chunk
        
        Returns:
            List of similar chunks with context
        """
        logger.info(f"Searching for similar chunks to query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Retrieve all chunks from the database
            conn = self.db_manager.conn
            cursor = conn.cursor()
            
            # Fetch all chunks with their embeddings
            cursor.execute('SELECT id, document_id, text, embedding FROM chunks')
            chunk_data = cursor.fetchall()
            logger.info(f"Total chunks retrieved: {len(chunk_data)}")
            
            # Compute similarities
            similarities = []
            for chunk_id, doc_id, chunk_text, chunk_embedding_blob in chunk_data:
                try:
                    # Convert embedding blob to numpy array
                    chunk_embedding = np.frombuffer(chunk_embedding_blob, dtype=np.float32)
                    
                    # Ensure consistent dimensionality
                    if len(chunk_embedding) > 384:
                        chunk_embedding = chunk_embedding[:384]
                    elif len(chunk_embedding) < 384:
                        chunk_embedding = np.pad(chunk_embedding, (0, 384 - len(chunk_embedding)), mode='constant')
                    
                    # Normalize chunk embedding
                    chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
                    
                    # Compute similarity (cosine similarity)
                    similarity = np.dot(chunk_embedding, query_embedding)
                    
                    similarities.append({
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'chunk_text': chunk_text,
                        'similarity': similarity
                    })
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
            
            # Sort similarities in descending order
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Select top k results
            top_results = similarities[:limit]
            
            logger.info(f"Top {len(top_results)} similar chunks:")
            for result in top_results:
                logger.info(f"Similarity: {result['similarity']:.4f}, Text: {result['chunk_text'][:100]}...")
            
            return top_results
        
        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            return []
    
    def _get_chunk_context(self, chunk: Dict[str, Any]) -> str:
        """Get additional context for a chunk.
        
        Args:
            chunk: Chunk information
        
        Returns:
            Contextual information string
        """
        context_parts = []
        
        # Add filename
        if chunk.get('filename'):
            context_parts.append(f"Source: {chunk['filename']}")
        
        # Add page or chunk information
        if chunk.get('chunk_index') is not None and chunk.get('total_chunks') is not None:
            context_parts.append(f"Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}")
        
        # Add similarity score
        if chunk.get('similarity'):
            context_parts.append(f"Relevance: {chunk['similarity']:.2%}")
        
        return " | ".join(context_parts)
