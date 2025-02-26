import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import time

class SQLiteDatabaseManager:
    def __init__(self, db_path: str = "data/chatbot.db"):
        """Initialize the database manager with SQLite.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create chunks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            text TEXT,
            embedding BLOB,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        ''')
        
        self.conn.commit()
    
    def store_document(self, filename: str, metadata: Dict[str, Any]) -> str:
        """Store document metadata in the database.
        
        Args:
            filename: Name of the document
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        cursor = self.conn.cursor()
        
        # Check if document already exists
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        existing_doc = cursor.fetchone()
        
        if existing_doc:
            return str(existing_doc[0])
        
        # Insert new document
        cursor.execute(
            "INSERT INTO documents (filename, metadata) VALUES (?, ?)",
            (filename, json.dumps(metadata))
        )
        
        self.conn.commit()
        return str(cursor.lastrowid)
    
    def store_chunk(self, document_id: str, text: str, embedding: List[float], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a text chunk with its embedding in the database.
        
        Args:
            document_id: ID of the parent document
            text: Text content of the chunk
            embedding: Vector embedding of the chunk
            metadata: Additional metadata for the chunk
            
        Returns:
            Chunk ID
        """
        cursor = self.conn.cursor()
        
        # Serialize the embedding and metadata
        embedding_blob = pickle.dumps(embedding)
        metadata_json = json.dumps(metadata or {})
        
        # Insert chunk
        cursor.execute(
            "INSERT INTO chunks (document_id, text, embedding, metadata) VALUES (?, ?, ?, ?)",
            (document_id, text, embedding_blob, metadata_json)
        )
        
        self.conn.commit()
        return str(cursor.lastrowid)
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query embedding.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            
        Returns:
            List of similar chunks with their documents
        """
        cursor = self.conn.cursor()
        
        # Get all chunks
        cursor.execute("SELECT id, document_id, text, embedding, metadata FROM chunks")
        all_chunks = cursor.fetchall()
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        # Calculate cosine similarity for each chunk
        results_with_scores = []
        for chunk_id, doc_id, text, embedding_blob, metadata_json in all_chunks:
            # Deserialize the embedding
            chunk_embedding = np.array(pickle.loads(embedding_blob))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding_np, chunk_embedding) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(chunk_embedding)
            )
            
            chunk = {
                "id": chunk_id,
                "document_id": doc_id,
                "text": text,
                "metadata": json.loads(metadata_json)
            }
            
            results_with_scores.append((chunk, similarity))
        
        # Sort by similarity score (descending)
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = results_with_scores[:limit]
        
        # Fetch document information for each chunk
        enriched_results = []
        for chunk, score in top_results:
            cursor.execute(
                "SELECT filename, metadata FROM documents WHERE id = ?", 
                (chunk["document_id"],)
            )
            doc_result = cursor.fetchone()
            
            if doc_result:
                filename, metadata_json = doc_result
                document = {
                    "_id": chunk["document_id"],
                    "filename": filename,
                    "metadata": json.loads(metadata_json)
                }
                
                enriched_results.append({
                    "chunk": chunk,
                    "document": document,
                    "similarity_score": score
                })
        
        return enriched_results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the database.
        
        Returns:
            List of documents
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename, metadata FROM documents")
        
        documents = []
        for doc_id, filename, metadata_json in cursor.fetchall():
            documents.append({
                "_id": doc_id,
                "filename": filename,
                "metadata": json.loads(metadata_json)
            })
        
        return documents
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Delete all chunks associated with the document
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            
            # Delete the document
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
