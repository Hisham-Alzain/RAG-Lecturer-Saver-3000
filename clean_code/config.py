from dataclasses import dataclass

@dataclass
class Config:
    """Central configuration for the RAG system."""

    # Chunking - optimized for better context
    PARENT_CHUNK_SIZE: int = 1200  # chars
    PARENT_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 600  # chars
    CHILD_OVERLAP: int = 75

    # Retrieval - increased for better coverage
    TOP_K: int = 7  # Return more sources for synthesis
    RETRIEVAL_K: int = 15  # Retrieve more candidates before reranking

    # Reranking
    ENABLE_RERANKING: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Faster, good quality
    RERANK_CANDIDATES: int = 25

    # Semantic Cache
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIM_THRESHOLD: float = 0.95  # Higher threshold for cache hits
    CACHE_MAX_ENTRIES: int = 100

    # Paths and Models
    CHROMA_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    
    LLM_PROVIDERS = {
    "Groq (Llama 3.3 70B) ⚡": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "name": "Groq",
        "get_key_url": "https://console.groq.com/keys",
        "notes": "Fastest inference, 14,400 req/day free",
    },
    "Cerebras (Llama 3.3 70B) 🚀": {
        "base_url": "https://api.cerebras.ai/v1",
        "model": "llama-3.3-70b",
        "name": "Cerebras",
        "get_key_url": "https://cloud.cerebras.ai/",
        "notes": "~1000 tok/sec, very fast",
    },
    "Groq (Llama 3.1 8B) 🆓": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "name": "Groq",
        "get_key_url": "https://console.groq.com/keys",
        "notes": "Fast & lightweight",
    },
    "OpenRouter (Llama 3.1 8B) 🆓": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "name": "OpenRouter",
        "get_key_url": "https://openrouter.ai/keys",
        "notes": "Free tier available",
    },
}