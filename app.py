# app.py
# Production-Ready Multilingual RAG Chat - University Tutor
# Features: Intelligent retrieval, strict grounding, Arabic/English support

import streamlit as st
import hashlib
import uuid
import re
import math
import time
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Third-party imports
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from chromadb.config import Settings
from pptx import Presentation
from pypdf import PdfReader
from openai import OpenAI


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """Central configuration for the RAG system."""
    
    # Chunking - optimized for better context
    PARENT_CHUNK_SIZE: int = 1200  # chars
    PARENT_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 600   # chars  
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


CONFIG = Config()


# LLM Provider configurations
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


# ============================================================================
# PAGE SETUP & STYLING
# ============================================================================
st.set_page_config(
    page_title="RAG Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 1.25rem;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 1rem;
    }
    
    .assistant-message {
        background: #f7f7f8;
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .assistant-message-rtl {
        background: #f7f7f8;
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        line-height: 1.9;
        font-size: 1.05rem;
        direction: rtl;
        text-align: right;
    }
    
    .source-card {
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    
    .source-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102,126,234,0.15);
    }
    
    .source-number {
        background: #667eea;
        color: white;
        min-width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .citation {
        background: #667eea;
        color: white;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 0.1rem;
    }
    
    .section-label {
        color: #888;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .app-header {
        text-align: center;
        padding: 2rem 0;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        color: #666;
        font-size: 1rem;
    }
    
    .provider-badge {
        background: #e8f4e8;
        color: #2d6a2d;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "chat_history": [],
        "api_key": "",
        "selected_provider": list(LLM_PROVIDERS.keys())[0],
        "semantic_cache": OrderedDict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================================
# LANGUAGE UTILITIES
# ============================================================================
class LanguageUtils:
    """Language detection and handling utilities."""
    
    ARABIC_RANGE = ("\u0600", "\u06ff")
    
    ARABIC_STOP_WORDS = {
        "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
        "التي", "الذي", "هو", "هي", "هم", "أن", "إن", "كان", "كانت",
        "ما", "لا", "لم", "لن", "قد", "كل", "بعض", "أي", "و", "أو",
        "ثم", "حتى", "إذا", "لو", "كما", "بل", "لكن", "غير", "بين",
        "عند", "منذ", "حول", "خلال", "بعد", "قبل", "فوق", "تحت",
        "ال", "الى", "هل", "كيف", "لماذا", "متى", "أين",
    }
    
    ENGLISH_STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "what",
        "which", "who", "when", "where", "why", "how", "this", "that",
        "these", "those", "it", "its", "as", "by", "from", "about",
        "into", "through", "can", "could", "would", "should", "will",
    }
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        if not text:
            return "en"
        
        arabic_chars = sum(1 for c in text if cls.ARABIC_RANGE[0] <= c <= cls.ARABIC_RANGE[1])
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return "en"
        
        return "ar" if (arabic_chars / total_alpha) > 0.3 else "en"
    
    @classmethod
    def get_stop_words(cls, lang: str) -> set:
        """Get stop words for the specified language."""
        return cls.ARABIC_STOP_WORDS if lang == "ar" else cls.ENGLISH_STOP_WORDS
    
    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        """Extract keywords from text, removing stop words."""
        lang = cls.detect_language(text)
        stop_words = cls.get_stop_words(lang)
        
        # Tokenize
        if lang == "ar":
            words = re.findall(r'[\u0600-\u06ff]+|\b\w+\b', text.lower())
        else:
            words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return unique keywords preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        
        return unique[:10]  # Max 10 keywords


# ============================================================================
# LLM CLIENT
# ============================================================================
class LLMClient:
    """LLM API client with provider-specific handling."""
    
    def __init__(self, provider_key: str, api_key: str):
        config = LLM_PROVIDERS[provider_key]
        self.model = config["model"]
        self.provider_name = config["name"]
        
        # OpenRouter requires additional headers
        headers = {}
        if "openrouter" in config["base_url"].lower():
            headers = {
                "HTTP-Referer": "https://rag-tutor.streamlit.app",
                "X-Title": "RAG Tutor",
            }
        
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=api_key,
            default_headers=headers if headers else None,
        )
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Send chat completion request."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


# ============================================================================
# MODEL LOADERS
# ============================================================================
@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load embedding model."""
    return SentenceTransformer(CONFIG.EMBEDDING_MODEL)


@st.cache_resource
def load_reranker() -> CrossEncoder:
    """Load reranker model."""
    return CrossEncoder(CONFIG.RERANK_MODEL)


def get_chroma_collection():
    """Get or create ChromaDB collection."""
    if "chroma_collection" not in st.session_state:
        client = PersistentClient(
            path=CONFIG.CHROMA_DIR,
            settings=Settings(allow_reset=True)
        )
        st.session_state["chroma_client"] = client
        st.session_state["chroma_collection"] = client.get_or_create_collection(
            "documents",
            metadata={"hnsw:space": "cosine"}
        )
    return st.session_state["chroma_client"], st.session_state["chroma_collection"]


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
class DocumentProcessor:
    """Document extraction and chunking."""
    
    @staticmethod
    def extract_from_pdf(file) -> List[Tuple[int, str]]:
        """Extract text from PDF."""
        try:
            reader = PdfReader(file)
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((i, text.strip()))
            return pages
        except Exception as e:
            st.error(f"PDF error: {e}")
            return []
    
    @staticmethod
    def extract_from_ppt(file) -> List[Tuple[int, str]]:
        """Extract text from PowerPoint."""
        try:
            prs = Presentation(file)
            slides = []
            for i, slide in enumerate(prs.slides, 1):
                texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        texts.append(shape.text.strip())
                if texts:
                    slides.append((i, "\n".join(texts)))
            return slides
        except Exception as e:
            st.error(f"PPT error: {e}")
            return []
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '.\n', '؟ ', '。', '\n\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    @classmethod
    def process_document(cls, file) -> List[Dict]:
        """Process a document into chunks with metadata."""
        filename = file.name.lower()
        
        if filename.endswith('.pdf'):
            pages = cls.extract_from_pdf(file)
            doc_type = "pdf"
        elif filename.endswith(('.ppt', '.pptx')):
            pages = cls.extract_from_ppt(file)
            doc_type = "ppt"
        else:
            return []
        
        all_chunks = []
        
        for page_num, text in pages:
            lang = LanguageUtils.detect_language(text)
            
            # Create parent chunks
            parent_chunks = cls.chunk_text(
                text, 
                CONFIG.PARENT_CHUNK_SIZE, 
                CONFIG.PARENT_OVERLAP
            )
            
            for p_idx, parent_text in enumerate(parent_chunks):
                # Create child chunks from parent
                child_chunks = cls.chunk_text(
                    parent_text,
                    CONFIG.CHILD_CHUNK_SIZE,
                    CONFIG.CHILD_OVERLAP
                )
                
                for c_idx, child_text in enumerate(child_chunks):
                    all_chunks.append({
                        "child_text": child_text,
                        "parent_text": parent_text,
                        "metadata": {
                            "source": file.name,
                            "page": page_num,
                            "type": doc_type,
                            "lang": lang,
                            "parent_idx": p_idx,
                            "child_idx": c_idx,
                        }
                    })
        
        return all_chunks


class DocumentIngester:
    """Ingest documents into vector database."""
    
    def __init__(self, embedder: SentenceTransformer, collection):
        self.embedder = embedder
        self.collection = collection
    
    @staticmethod
    def get_file_hash(file) -> str:
        """Generate file hash."""
        file.seek(0)
        h = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        return h
    
    def is_ingested(self, file_hash: str) -> bool:
        """Check if file already ingested."""
        try:
            results = self.collection.get(where={"file_hash": file_hash}, limit=1)
            return len(results["ids"]) > 0
        except:
            return False
    
    def ingest(self, file) -> Tuple[int, str]:
        """Ingest a file into the database."""
        file_hash = self.get_file_hash(file)
        
        if self.is_ingested(file_hash):
            return 0, "exists"
        
        chunks = DocumentProcessor.process_document(file)
        if not chunks:
            return 0, "no content"
        
        # Prepare for embedding
        texts_to_embed = []
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{i}"
            
            # Format for E5 model
            embed_text = f"passage: {chunk['child_text']}"
            texts_to_embed.append(embed_text)
            
            ids.append(chunk_id)
            documents.append(chunk["child_text"])
            metadatas.append({
                **chunk["metadata"],
                "file_hash": file_hash,
                "parent_text": chunk["parent_text"],
            })
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            texts_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        return len(chunks), "success"


# ============================================================================
# INTELLIGENT RETRIEVER
# ============================================================================
class IntelligentRetriever:
    """
    Smart retrieval with query understanding and multi-strategy search.
    
    Key features:
    1. Query analysis - understands question type (factual, comparison, etc.)
    2. Query expansion - generates related queries for better coverage
    3. Multi-vector search - searches with multiple query formulations
    4. Cross-encoder reranking - ensures relevance
    5. Parent context retrieval - returns richer context
    """
    
    def __init__(self, embedder: SentenceTransformer, collection, llm_client: LLMClient):
        self.embedder = embedder
        self.collection = collection
        self.llm_client = llm_client
        self.reranker = None
        
        if CONFIG.ENABLE_RERANKING:
            try:
                self.reranker = load_reranker()
            except Exception as e:
                st.warning(f"Reranker not loaded: {e}")
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to understand what information is needed.
        Returns query type and extracted entities/topics.
        """
        query_lower = query.lower()
        lang = LanguageUtils.detect_language(query)
        
        analysis = {
            "lang": lang,
            "type": "factual",  # default
            "topics": [],
            "is_comparison": False,
            "is_definition": False,
            "is_explanation": False,
            "is_list": False,
        }
        
        # Detect comparison questions
        comparison_patterns_en = [
            r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+)",
            r"difference(?:s)?\s+between\s+(.+?)\s+(?:and|&)\s+(.+)",
            r"(.+?)\s+vs\.?\s+(.+)",
            r"contrast\s+(.+?)\s+(?:and|with)\s+(.+)",
            r"how\s+(?:does|do|is|are)\s+(.+?)\s+differ\s+from\s+(.+)",
        ]
        comparison_patterns_ar = [
            r"قارن\s+(?:بين\s+)?(.+?)\s+(?:و|مع)\s+(.+)",
            r"الفرق\s+بين\s+(.+?)\s+(?:و|&)\s+(.+)",
            r"ما\s+(?:هو\s+)?الفرق\s+بين\s+(.+?)\s+(?:و|&)\s+(.+)",
            r"مقارنة\s+(?:بين\s+)?(.+?)\s+(?:و|مع)\s+(.+)",
        ]
        
        patterns = comparison_patterns_ar if lang == "ar" else comparison_patterns_en
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                analysis["is_comparison"] = True
                analysis["type"] = "comparison"
                topic1 = match.group(1).strip().strip('?.,')
                topic2 = match.group(2).strip().strip('?.,')
                # Clean articles
                topic1 = re.sub(r'^(the|a|an)\s+', '', topic1, flags=re.IGNORECASE)
                topic2 = re.sub(r'^(the|a|an)\s+', '', topic2, flags=re.IGNORECASE)
                if topic1 and topic2:
                    analysis["topics"] = [topic1, topic2]
                break
        
        # Detect definition questions
        if not analysis["is_comparison"]:
            definition_patterns = [
                r"what\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)",
                r"define\s+(.+)",
                r"ما\s+(?:هو|هي)\s+(.+)",
                r"تعريف\s+(.+)",
                r"عرف\s+(.+)",
            ]
            for pattern in definition_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    analysis["is_definition"] = True
                    analysis["type"] = "definition"
                    topic = match.group(1).strip().strip('?.,')
                    if topic:
                        analysis["topics"] = [topic]
                    break
        
        # Detect explanation questions
        if not analysis["topics"]:
            explanation_patterns = [
                r"how\s+(?:does|do|can|to)\s+(.+)",
                r"explain\s+(.+)",
                r"why\s+(?:does|do|is|are)\s+(.+)",
                r"كيف\s+(.+)",
                r"اشرح\s+(.+)",
                r"لماذا\s+(.+)",
            ]
            for pattern in explanation_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    analysis["is_explanation"] = True
                    analysis["type"] = "explanation"
                    topic = match.group(1).strip().strip('?.,')
                    if topic:
                        analysis["topics"] = [topic]
                    break
        
        # Detect list questions
        list_patterns = [
            r"(?:what|list|name)\s+(?:are\s+)?(?:the\s+)?(?:different\s+)?(?:types|kinds|methods|ways|steps)",
            r"ما\s+(?:هي\s+)?(?:أنواع|طرق|خطوات)",
        ]
        for pattern in list_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                analysis["is_list"] = True
                analysis["type"] = "list"
                break
        
        # Extract keywords if no topics found
        if not analysis["topics"]:
            analysis["topics"] = LanguageUtils.extract_keywords(query)[:5]
        
        return analysis
    
    def generate_search_queries(self, query: str, analysis: Dict) -> List[str]:
        """Generate multiple search queries for comprehensive retrieval."""
        queries = [query]  # Always include original
        lang = analysis["lang"]
        
        # For comparison questions, search each topic separately
        if analysis["is_comparison"] and len(analysis["topics"]) >= 2:
            for topic in analysis["topics"]:
                if lang == "ar":
                    queries.extend([
                        f"ما هو {topic}",
                        f"تعريف {topic}",
                        topic,
                    ])
                else:
                    queries.extend([
                        f"what is {topic}",
                        f"{topic} definition",
                        f"{topic} explanation",
                        topic,
                    ])
        
        # For definition questions
        elif analysis["is_definition"] and analysis["topics"]:
            topic = analysis["topics"][0]
            if lang == "ar":
                queries.extend([
                    topic,
                    f"تعريف {topic}",
                    f"شرح {topic}",
                ])
            else:
                queries.extend([
                    topic,
                    f"{topic} definition",
                    f"{topic} meaning",
                    f"define {topic}",
                ])
        
        # For other questions, add keyword-based queries
        else:
            keywords = analysis["topics"][:3]
            if keywords:
                queries.append(" ".join(keywords))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:8]  # Max 8 queries
    
    def vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search."""
        # Format for E5 model
        query_embedding = self.embedder.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        return [
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,  # Convert distance to similarity
            }
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    
    def merge_results(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Merge results from multiple queries using RRF."""
        scores = defaultdict(float)
        doc_data = {}
        k = 60  # RRF constant
        
        for results in results_list:
            for rank, result in enumerate(results):
                doc_id = result["id"]
                scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            {**doc_data[doc_id], "rrf_score": scores[doc_id]}
            for doc_id in sorted_ids
            if doc_id in doc_data
        ]
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if not self.reranker or not candidates:
            return candidates[:top_k]
        
        # Prepare pairs for reranking
        pairs = []
        for c in candidates[:CONFIG.RERANK_CANDIDATES]:
            # Use parent text for richer context
            text = c["metadata"].get("parent_text", c["document"])
            # Truncate if too long
            if len(text) > 512:
                text = text[:512]
            pairs.append([query, text])
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for i, c in enumerate(candidates[:len(scores)]):
            c["rerank_score"] = float(scores[i])
        
        candidates[:len(scores)] = sorted(
            candidates[:len(scores)],
            key=lambda x: x.get("rerank_score", -float("inf")),
            reverse=True
        )
        
        return candidates[:top_k]
    
    def deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks from same parent."""
        seen_parents = set()
        unique = []
        
        for r in results:
            parent_id = f"{r['metadata'].get('source', '')}_{r['metadata'].get('page', '')}_{r['metadata'].get('parent_idx', '')}"
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique.append(r)
        
        return unique
    
    def retrieve(self, query: str) -> Tuple[List[Dict], Dict]:
        """
        Main retrieval pipeline.
        Returns (results, debug_info)
        """
        debug = {
            "query_type": "unknown",
            "topics": [],
            "num_queries": 0,
            "num_candidates": 0,
        }
        
        # Step 1: Analyze query
        analysis = self.analyze_query(query)
        debug["query_type"] = analysis["type"]
        debug["topics"] = analysis["topics"]
        
        # Step 2: Generate search queries
        search_queries = self.generate_search_queries(query, analysis)
        debug["num_queries"] = len(search_queries)
        
        # Step 3: Search with each query
        all_results = []
        for q in search_queries:
            results = self.vector_search(q, CONFIG.RETRIEVAL_K)
            if results:
                all_results.append(results)
        
        if not all_results:
            return [], debug
        
        # Step 4: Merge results
        merged = self.merge_results(all_results)
        debug["num_candidates"] = len(merged)
        
        # Step 5: Deduplicate
        deduped = self.deduplicate(merged)
        
        # Step 6: Rerank
        reranked = self.rerank(query, deduped, CONFIG.TOP_K)
        
        return reranked, debug


# ============================================================================
# ANSWER GENERATOR
# ============================================================================
class AnswerGenerator:
    """
    Generate grounded answers using retrieved context.
    
    Key principles:
    1. ONLY use information from provided sources
    2. ALWAYS cite sources for factual claims
    3. If information not found, say so clearly
    4. Can synthesize/compare if info exists in sources
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def get_system_prompt(self, lang: str) -> str:
        """Get system prompt based on language."""
        if lang == "ar":
            return """أنت مساعد تعليمي ذكي. مهمتك هي الإجابة على أسئلة الطلاب باستخدام المصادر المقدمة فقط.

## القواعد الأساسية:

### 1. استخدم المصادر فقط
- أجب فقط باستخدام المعلومات الموجودة في المصادر المقدمة
- لا تستخدم معرفتك العامة أبداً
- إذا لم تجد المعلومة: قل "هذه المعلومات غير متوفرة في المواد المقدمة"

### 2. الاستشهاد إلزامي
- ضع [رقم] بعد كل معلومة مباشرة
- مثال: "RGB هو نموذج لوني [1]"
- لا تذكر معلومة بدون استشهاد

### 3. أسئلة المقارنة
- إذا طُلب مقارنة موضوعين:
  - ابحث عن كل موضوع في المصادر
  - إذا وجدت كليهما: قارن بينهما مع الاستشهاد لكل معلومة
  - إذا وجدت أحدهما فقط: اشرح ما وجدته واذكر أن الآخر غير موجود
  - إذا لم تجد أياً منهما: قل ذلك بوضوح

### 4. التنسيق
- اكتب بالعربية الفصحى
- المصطلحات التقنية: بالعربية ثم (English)
- فقرات واضحة ومترابطة
- 100-200 كلمة تقريباً"""
        
        else:
            return """You are an intelligent tutoring assistant. Your task is to answer student questions using ONLY the provided sources.

## Core Rules:

### 1. Use Sources Only
- Answer ONLY using information from the provided sources
- NEVER use your general knowledge
- If information not found: Say "This information is not available in the provided materials"

### 2. Citations are Mandatory
- Add [number] after every factual statement
- Example: "RGB is a color model [1]"
- Never state a fact without citation

### 3. Comparison Questions
- If asked to compare two topics:
  - Search for each topic in the sources
  - If BOTH found: Compare them with citations for each fact
  - If only ONE found: Explain what you found and state the other is not in the materials
  - If NEITHER found: State this clearly

### 4. Synthesis is Allowed
- You CAN combine information from different sources
- You CAN explain relationships between concepts if the individual concepts are in sources
- You CAN use reasoning to answer based on source content
- But EVERY fact must be cited

### 5. Format
- Clear, flowing paragraphs
- Academic but accessible
- About 100-200 words"""
    
    def build_context(self, retrieved_docs: List[Dict]) -> Tuple[str, List[Dict]]:
        """Build context string and source list from retrieved documents."""
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Use parent text for richer context
            text = doc["metadata"].get("parent_text", doc["document"])
            context_parts.append(f"[Source {i}]:\n{text}")
            
            sources.append({
                "source": doc["metadata"].get("source", "Unknown"),
                "page": doc["metadata"].get("page", "?"),
                "lang": doc["metadata"].get("lang", "en"),
                "text": text,
            })
        
        return "\n\n".join(context_parts), sources
    
    def generate(self, query: str, retrieved_docs: List[Dict]) -> Tuple[str, List[Dict]]:
        """Generate answer from retrieved documents."""
        lang = LanguageUtils.detect_language(query)
        
        if not retrieved_docs:
            no_info = (
                "لم أجد معلومات ذات صلة في المواد المرفوعة. يرجى التأكد من رفع الملفات المناسبة."
                if lang == "ar"
                else "I couldn't find relevant information in the uploaded materials. Please make sure to upload the appropriate files."
            )
            return no_info, []
        
        # Build context
        context, sources = self.build_context(retrieved_docs)
        
        # Get prompts
        system_prompt = self.get_system_prompt(lang)
        
        if lang == "ar":
            user_prompt = f"""المصادر المتاحة:
{context}

سؤال الطالب: {query}

تذكر:
- استخدم المصادر فقط
- استشهد بكل معلومة [رقم]
- إذا المعلومة غير موجودة، قل ذلك"""
        else:
            user_prompt = f"""Available Sources:
{context}

Student Question: {query}

Remember:
- Use ONLY the sources above
- Cite every fact [number]
- If information is not found, say so clearly"""
        
        # Generate answer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        answer = self.llm_client.chat(messages, temperature=0.3, max_tokens=1024)
        
        # Add sources section
        if lang == "ar":
            sources_header = "\n\nالمصادر:"
            source_lines = [f"[{i}] {s['source']} - صفحة {s['page']}" for i, s in enumerate(sources, 1)]
        else:
            sources_header = "\n\nSources:"
            source_lines = [f"[{i}] {s['source']} - page {s['page']}" for i, s in enumerate(sources, 1)]
        
        answer = answer.strip() + sources_header + "\n" + "\n".join(source_lines)
        
        return answer, sources
    
    def generate_followups(self, query: str, answer: str, lang: str) -> List[str]:
        """Generate follow-up questions."""
        if lang == "ar":
            prompt = """بناءً على السؤال والجواب، اقترح 3 أسئلة متابعة قصيرة بالعربية.
اكتب الأسئلة فقط، سؤال في كل سطر، بدون ترقيم.

السؤال: {query}
الجواب: {answer}"""
        else:
            prompt = """Based on this Q&A, suggest 3 short follow-up questions in English.
Write only the questions, one per line, without numbering.

Question: {query}
Answer: {answer}"""
        
        try:
            result = self.llm_client.chat(
                [{"role": "user", "content": prompt.format(query=query, answer=answer[:500])}],
                temperature=0.7,
                max_tokens=200,
            )
            
            lines = [l.strip() for l in result.split("\n") if l.strip()]
            questions = []
            for line in lines[:3]:
                # Clean numbering
                line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
                if line and len(line) > 10:
                    questions.append(line)
            return questions
        except:
            return []


# ============================================================================
# SEMANTIC CACHE
# ============================================================================
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
        if not CONFIG.ENABLE_SEMANTIC_CACHE:
            return None
        
        best_match = None
        best_sim = -1
        
        for key, value in self.cache.items():
            cached_emb = value.get("embedding")
            if cached_emb:
                sim = self._cosine_sim(query_embedding, cached_emb)
                if sim > best_sim and sim >= CONFIG.CACHE_SIM_THRESHOLD:
                    best_sim = sim
                    best_match = value
        
        return best_match
    
    def store(self, query: str, embedding: List[float], answer: str, sources: List[Dict], followups: List[str]):
        """Store query result in cache."""
        if not CONFIG.ENABLE_SEMANTIC_CACHE:
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
        while len(self.cache) > CONFIG.CACHE_MAX_ENTRIES:
            self.cache.popitem(last=False)


# ============================================================================
# RAG PIPELINE
# ============================================================================
class RAGPipeline:
    """Main RAG pipeline."""
    
    def __init__(self, llm_client: LLMClient, embedder: SentenceTransformer, collection):
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
            [f"query: {query}"],
            normalize_embeddings=True
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


# ============================================================================
# UI COMPONENTS
# ============================================================================
class UI:
    """UI rendering components."""
    
    @staticmethod
    def render_sources(sources: List[Dict]):
        """Render source cards."""
        if not sources:
            return
        
        st.markdown('<div class="section-label">📚 Sources</div>', unsafe_allow_html=True)
        
        cols = st.columns(min(len(sources), 3))
        for i, (col, src) in enumerate(zip(cols * ((len(sources) // 3) + 1), sources)):
            with cols[i % 3]:
                name = src.get("source", "Unknown")
                page = src.get("page", "?")
                display_name = name[:25] + "..." if len(name) > 25 else name
                
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-number">{i + 1}</div>
                    <div>
                        <div style="font-weight: 500; font-size: 0.85rem;">{display_name}</div>
                        <div style="color: #666; font-size: 0.75rem;">Page {page}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_answer(answer: str, is_rtl: bool = False):
        """Render answer with styled citations."""
        def replace_citation(match):
            return f'<span class="citation">{match.group(1)}</span>'
        
        styled = re.sub(r'\[(\d+)\]', replace_citation, answer)
        styled = styled.replace('\n\n', '</p><p>').replace('\n', '<br>')
        styled = f'<p>{styled}</p>'
        
        css_class = "assistant-message-rtl" if is_rtl else "assistant-message"
        st.markdown(f'<div class="{css_class}">{styled}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_followups(questions: List[str], lang: str, msg_idx: int) -> Optional[str]:
        """Render follow-up question buttons."""
        if not questions:
            return None
        
        label = "أسئلة ذات صلة" if lang == "ar" else "Related Questions"
        st.markdown(f'<div class="section-label">💡 {label}</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(questions))
        for i, (col, q) in enumerate(zip(cols, questions)):
            with col:
                if st.button(q, key=f"followup_{msg_idx}_{i}", use_container_width=True):
                    return q
        return None
    
    @classmethod
    def render_message(cls, role: str, content: str, sources: List[Dict] = None, 
                      followups: List[str] = None, lang: str = "en", msg_idx: int = 0) -> Optional[str]:
        """Render a chat message."""
        if role == "user":
            is_rtl = LanguageUtils.detect_language(content) == "ar"
            direction = 'dir="rtl"' if is_rtl else ""
            st.markdown(f'<div class="user-message" {direction}>{content}</div>', unsafe_allow_html=True)
        else:
            if sources:
                cls.render_sources(sources)
            cls.render_answer(content, LanguageUtils.detect_language(content) == "ar")
            if followups:
                return cls.render_followups(followups, lang, msg_idx)
        return None


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    """Render sidebar with settings and upload."""
    with st.sidebar:
        st.markdown("### ⚙️ LLM Provider")
        
        selected = st.selectbox(
            "Provider",
            list(LLM_PROVIDERS.keys()),
            index=list(LLM_PROVIDERS.keys()).index(st.session_state.selected_provider),
            label_visibility="collapsed",
        )
        st.session_state.selected_provider = selected
        
        config = LLM_PROVIDERS[selected]
        st.caption(f"ℹ️ {config['notes']}")
        
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder=f"Enter {config['name']} key",
        )
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown(f"[🔑 Get API key]({config['get_key_url']})")
        
        st.markdown("---")
        st.markdown("### 📄 Documents")
        
        files = st.file_uploader(
            "Upload",
            type=["pdf", "ppt", "pptx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        
        if st.button("📥 Ingest", use_container_width=True):
            if not files:
                st.error("Upload files first")
            else:
                embedder = load_embedder()
                _, collection = get_chroma_collection()
                ingester = DocumentIngester(embedder, collection)
                
                total = 0
                for f in files:
                    with st.spinner(f"Processing {f.name}..."):
                        count, status = ingester.ingest(f)
                        if status == "exists":
                            st.info(f"✓ {f.name} already indexed")
                        elif status == "success":
                            st.success(f"✓ {f.name}: {count} chunks")
                            total += count
                        else:
                            st.error(f"✗ {f.name}: {status}")
                
                if total > 0:
                    st.success(f"✅ Added {total} chunks")
        
        try:
            _, collection = get_chroma_collection()
            count = collection.count()
            st.caption(f"📊 {count} chunks indexed")
        except:
            pass
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset DB", use_container_width=True):
                try:
                    client, _ = get_chroma_collection()
                    client.reset()
                    for key in ["chroma_client", "chroma_collection"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.semantic_cache = OrderedDict()
                    st.success("Reset!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main application."""
    render_sidebar()
    
    # Welcome
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="app-header">
            <div class="app-title">🎓 RAG Tutor</div>
            <div class="app-subtitle">Ask questions about your lecture materials in English or Arabic</div>
        </div>
        """, unsafe_allow_html=True)
        
        provider = LLM_PROVIDERS[st.session_state.selected_provider]["name"]
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="provider-badge">⚡ Powered by {provider}</span>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.api_key:
        st.info("👈 Enter your API key in the sidebar to get started")
        st.stop()
    
    # Initialize client
    llm_client = LLMClient(st.session_state.selected_provider, st.session_state.api_key)
    
    # Render history
    followup_clicked = None
    for i, msg in enumerate(st.session_state.chat_history):
        result = UI.render_message(
            msg["role"],
            msg["content"],
            msg.get("sources"),
            msg.get("followups"),
            msg.get("lang", "en"),
            i,
        )
        if result:
            followup_clicked = result
    
    if followup_clicked:
        st.session_state.pending_question = followup_clicked
        st.rerun()
    
    # Input
    question = st.chat_input("Ask about your materials... / اسأل عن موادك...")
    
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
    
    if question:
        question = question.strip()
        if question:
            lang = LanguageUtils.detect_language(question)
            st.session_state.chat_history.append({"role": "user", "content": question})
            UI.render_message("user", question)
            
            with st.spinner("🔍 Searching and analyzing..."):
                embedder = load_embedder()
                _, collection = get_chroma_collection()
                
                pipeline = RAGPipeline(llm_client, embedder, collection)
                answer, sources, followups, debug = pipeline.answer(question)
                
                # Debug info
                info_parts = []
                if debug.get("cache_hit"):
                    info_parts.append("Cache hit")
                if debug.get("query_type"):
                    info_parts.append(f"Type: {debug['query_type']}")
                if debug.get("topics"):
                    info_parts.append(f"Topics: {', '.join(debug['topics'][:3])}")
                
                if info_parts:
                    st.caption(f"🔧 {' • '.join(info_parts)}")
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "followups": followups,
                "lang": lang,
            })
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <strong>Features:</strong> Intelligent Query Analysis • Multi-Strategy Retrieval • 
        Cross-Encoder Reranking • Strict Grounding • Bilingual Support
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()