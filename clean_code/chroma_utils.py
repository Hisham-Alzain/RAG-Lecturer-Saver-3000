import streamlit as st
from chromadb import PersistentClient
from chromadb.config import Settings
from config import Config


def get_chroma_collection():
    """Get or create ChromaDB collection."""
    if "chroma_collection" not in st.session_state:
        client = PersistentClient(
            path=Config.CHROMA_DIR, settings=Settings(allow_reset=True)
        )
        st.session_state["chroma_client"] = client
        st.session_state["chroma_collection"] = client.get_or_create_collection(
            "documents", metadata={"hnsw:space": "cosine"}
        )
    return st.session_state["chroma_client"], st.session_state["chroma_collection"]