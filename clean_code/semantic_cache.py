import numpy as np
import hashlib
from collections import OrderedDict
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from config import Config

class SemanticCache:
    """Cache for semantically similar queries."""

    def __init__(self, cache: OrderedDict, embedder: SentenceTransformer):
        self.cache = cache
        self.embedder = embedder

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b))

    def lookup(self, query: str, query_embedding: List[float]) -> Optional[Dict]:
        """Look for semantically similar cached query."""
        if not Config.ENABLE_SEMANTIC_CACHE:
            return None

        best_match = None
        best_sim = -1

        for key, value in self.cache.items():
            cached_emb = value.get("embedding")
            if cached_emb:
                sim = self._cosine_sim(query_embedding, cached_emb)
                if sim > best_sim and sim >= Config.CACHE_SIM_THRESHOLD:
                    best_sim = sim
                    best_match = value

        return best_match

    def store(
        self,
        query: str,
        embedding: List[float],
        answer: str,
        sources: List[Dict],
        followups: List[str],
    ):
        """Store query result in cache."""
        if not Config.ENABLE_SEMANTIC_CACHE:
            return

        key = hashlib.md5(query.encode()).hexdigest()

        self.cache[key] = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "sources": sources,
            "followups": followups,
        }

        # LRU eviction
        while len(self.cache) > Config.CACHE_MAX_ENTRIES:
            self.cache.popitem(last=False)