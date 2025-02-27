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
    
    def generate_response(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response with optional context.
        
        Args:
            query: User's query or enhanced prompt
            context: Optional context chunks for more informed responses
        
        Returns:
            Generated response as a string
        """
        # Prepare context information
        context_info = ""
        if context:
            context_info = "\n\nContext Sources:\n" + "\n".join([
                f"- {chunk.get('filename', 'Unknown Source')} (Relevance: {chunk.get('similarity', 0):.2f})"
                for chunk in context
            ])
        
        # Combine query with context
        full_prompt = f"{query}{context_info}"
        
        try:
            # Generate response using LLaMA model
            response = self.llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant trained to provide accurate and contextual responses."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # Extract and clean the response
            generated_text = response['choices'][0]['message']['content'].strip()
            
            return generated_text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
