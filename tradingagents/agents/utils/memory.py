import os
import faiss
import numpy as np
import openai
import google.generativeai as genai
from typing import Dict, Any, List

# We will get the config directly when the class is initialized
# from tradingagents.config import config 

class FinancialSituationMemory:
    """A memory system for financial agents using FAISS for similarity search."""

    def __init__(self, name: str, config: Dict[str, Any], embedding_model: str = "text-embedding-3-small"):
        self.name = name
        self.config = config
        self.llm_provider = self.config.get("llm_provider", "openai").lower()

        # Configure the client and embedding settings based on the LLM provider
        if self.llm_provider == 'openai':
            self.embedding_model = embedding_model
            self.client = openai.OpenAI()
            # OpenAI's text-embedding-3-small has 1536 dimensions
            self.index = faiss.IndexFlatL2(1536)
        elif self.llm_provider == 'google':
            self.embedding_model = "models/embedding-001"
            # Ensure the API key is configured for the genai library
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = None  # Not needed for google's module-level functions
            # Google's embedding-001 model has 768 dimensions
            self.index = faiss.IndexFlatL2(768)
        else:
            raise ValueError(f"Unsupported LLM provider for embeddings: {self.llm_provider}")

        self.memory_vectors = []
        self.texts = []

    def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using the configured provider."""
        text = str(text).replace("\n", " ")
        
        if self.llm_provider == 'openai':
            response = self.client.embeddings.create(input=[text], model=self.embedding_model)
            return response.data[0].embedding
        
        elif self.llm_provider == 'google':
            result = genai.embed_content(model=self.embedding_model, content=text)
            return result['embedding']

    def add_memory(self, memory_text: str):
        """Adds a memory to the store."""
        if not memory_text or not memory_text.strip():
            return
            
        embedding = self.get_embedding(memory_text)
        if embedding:
            embedding_np = np.array([embedding], dtype='float32')
            self.index.add(embedding_np)
            self.texts.append(memory_text)

    def get_memories(self, current_situation: str, n_matches: int = 1) -> str:
        """Retrieves the most relevant memories for a given situation."""
        if self.index.ntotal == 0:
            return "No past memories found."

        query_embedding = np.array([self.get_embedding(current_situation)], dtype='float32')
        distances, indices = self.index.search(query_embedding, k=min(n_matches, self.index.ntotal))

        recalled_memories = []
        for i in indices[0]:
            if i != -1:
                recalled_memories.append(self.texts[i])
        
        return "\n\n".join(recalled_memories)

    def clear_memory(self):
        """Clears all memories from the store."""
        self.index.reset()
        self.memory_vectors = []
        self.texts = []