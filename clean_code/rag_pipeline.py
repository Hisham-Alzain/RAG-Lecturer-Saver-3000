import streamlit as st
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer
from intelligent_retriever import IntelligentRetriever
from answer_generator import AnswerGenerator
from semantic_cache import SemanticCache
from languageUtils import LanguageUtils
from typing import List, Dict, Tuple
class RAGPipeline:
    """Main RAG pipeline."""

    def __init__(
        self, llm_client: LLMClient, embedder: SentenceTransformer, collection
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.retriever = IntelligentRetriever(embedder, collection, llm_client)
        self.generator = AnswerGenerator(llm_client)
        self.cache = SemanticCache(st.session_state.semantic_cache, embedder)

    def answer(self, query: str) -> Tuple[str, List[Dict], List[str], Dict]:
        """Process query and return answer, sources, followups, debug info."""
        lang = LanguageUtils.detect_language(query)
        debug = {"cache_hit": False}

        # Get query embedding
        query_emb = self.embedder.encode(
            [f"query: {query}"], normalize_embeddings=True
        ).tolist()[0]

        # Check cache
        cached = self.cache.lookup(query, query_emb)
        if cached:
            debug["cache_hit"] = True
            return cached["answer"], cached["sources"], cached["followups"], debug

        # Retrieve
        retrieved, retrieval_debug = self.retriever.retrieve(query)
        debug.update(retrieval_debug)

        # Generate answer
        answer, sources = self.generator.generate(query, retrieved)

        # Generate followups
        followups = self.generator.generate_followups(query, answer, lang)

        # Cache result
        self.cache.store(query, query_emb, answer, sources, followups)

        return answer, sources, followups, debug