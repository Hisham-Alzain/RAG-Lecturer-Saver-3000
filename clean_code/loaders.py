import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import Config

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load embedding model."""
    return SentenceTransformer(Config.EMBEDDING_MODEL)


@st.cache_resource
def load_reranker() -> CrossEncoder:
    """Load reranker model."""
    return CrossEncoder(Config.RERANK_MODEL)