import os
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
import json

class LlamaModel:
    def __init__(self, model_path: str, 
                 context_size: int = 2048,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 n_gpu_layers: int = -1,
                 n_threads: Optional[int] = None):
        """Initialize the Llama model.
        
        Args:
            model_path: Path to the Llama model file
            context_size: Maximum context size
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of threads to use (None for auto)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize the model
        self.llama = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads
        )
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response for the given query using the provided context.
        
        Args:
            query: User query
            context: List of context chunks
            
        Returns:
            Generated response
        """
        # Format context
        formatted_context = self._format_context(context)
        
        # Create prompt
        prompt = self._create_prompt(query, formatted_context)
        
        # Generate response
        response = self.llama.create_completion(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["Human:", "USER:"],
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context chunks into a string.
        
        Args:
            context: List of context chunks
            
        Returns:
            Formatted context string
        """
        formatted_chunks = []
        
        for item in context:
            chunk = item["chunk"]
            document = item["document"]
            score = item["similarity_score"]
            
            formatted_chunk = f"Source: {document['filename']}\n"
            formatted_chunk += f"Relevance: {score:.4f}\n"
            formatted_chunk += f"Content: {chunk['text']}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        return "\n---\n".join(formatted_chunks)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the Llama model.
        
        Args:
            query: User query
            context: Formatted context
            
        Returns:
            Complete prompt
        """
        return f"""You are a helpful AI assistant that answers questions based on the provided context information. 
If the information needed to answer the question is not present in the context, say "I don't have enough information to answer that question."
Do not make up or infer information that is not directly supported by the context.
Always cite your sources when possible.

Context information:
{context}

Question: {query}

Answer:"""
