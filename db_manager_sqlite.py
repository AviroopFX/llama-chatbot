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
        """Enhanced method to search for similar chunks with more context.
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of chunks to return
        
        Returns:
            List of similar chunks with additional metadata
        """
        import numpy as np
        
        # Ensure query embedding is the right shape
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        if len(query_embedding) > 384:
            query_embedding = query_embedding[:384]
        elif len(query_embedding) < 384:
            query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), mode='constant')
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / query_norm if query_norm > 0 else query_embedding
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Retrieve all chunks with their embeddings
        cursor.execute("""
            SELECT 
                id, 
                document_id, 
                text,
                embedding,
                (SELECT filename FROM documents WHERE id = chunks.document_id) as filename
            FROM chunks
        """)
        
        chunks = []
        for row in cursor.fetchall():
            chunk_id, doc_id, text, embedding_blob, filename = row
            
            # Deserialize embedding
            try:
                # Convert blob to numpy array
                chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Ensure consistent dimensionality
                if chunk_embedding.ndim > 1:
                    chunk_embedding = chunk_embedding.flatten()
                if len(chunk_embedding) > 384:
                    chunk_embedding = chunk_embedding[:384]
                elif len(chunk_embedding) < 384:
                    chunk_embedding = np.pad(chunk_embedding, (0, 384 - len(chunk_embedding)), mode='constant')
                
                # Normalize chunk embedding
                chunk_norm = np.linalg.norm(chunk_embedding)
                chunk_embedding = chunk_embedding / chunk_norm if chunk_norm > 0 else chunk_embedding
                
                # Compute cosine similarity
                similarity = np.dot(chunk_embedding, query_embedding)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'document_id': doc_id,
                    'chunk_text': text,
                    'filename': filename,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
        
        conn.close()
        
        # Sort and limit results
        chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return chunks[:limit]
    
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
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its filename.
        
        Args:
            filename: Name of the file to search for
        
        Returns:
            Document information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, filename, metadata FROM documents WHERE filename = ?", (filename,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'filename': result[1],
                'metadata': json.loads(result[2]) if result[2] else {}
            }
        return None
    
    def delete_document(self, document_id: int):
        """Delete a document and its associated chunks.
        
        Args:
            document_id: ID of the document to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete associated chunks
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        
        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        
        conn.commit()
        conn.close()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
