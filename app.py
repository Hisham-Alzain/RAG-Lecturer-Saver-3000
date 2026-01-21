# app.py
# Enhanced Multilingual RAG Chat - University Tutor
# Refactored for better organization, Arabic/English support, and maintainability

import streamlit as st
import hashlib
import uuid
import re
import math
import time
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Third-party imports
from sentence-transformers import SentenceTransformer, CrossEncoder
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

    # Chunking
    PARENT_CHUNK_SIZE: int = 1500
    PARENT_OVERLAP: int = 200
    CHILD_CHUNK_SIZE: int = 512
    CHILD_OVERLAP: int = 50

    # Retrieval
    TOP_K: int = 5
    MULTI_QUERY_COUNT: int = 3

    # Reranking
    ENABLE_RERANKING: bool = True
    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANK_CANDIDATES: int = 30
    RERANK_TOP_K: int = 5

    # Semantic Cache
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIM_THRESHOLD: float = 0.92
    CACHE_MAX_ENTRIES: int = 200

    # CRAG (Corrective RAG)
    ENABLE_CRAG: bool = True
    CRAG_MIN_MAX_RELEVANCE: float = 0.35
    CRAG_MIN_MEAN_TOP3: float = 0.28
    CRAG_RETRIEVE_EXPANSION: int = 4

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
    "Groq (Llama 4 Scout) 🆕": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Groq",
        "get_key_url": "https://console.groq.com/keys",
        "notes": "Latest Llama 4, multimodal capable",
    },
    "OpenRouter (Llama 3.1 8B) 🆓": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "name": "OpenRouter",
        "get_key_url": "https://openrouter.ai/keys",
        "notes": "30+ free models available",
    },
    "OpenRouter (DeepSeek R1) 🧠": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-r1-0528:free",
        "name": "OpenRouter",
        "get_key_url": "https://openrouter.ai/keys",
        "notes": "Strong reasoning model",
    },
}


# ============================================================================
# PAGE SETUP & STYLING
# ============================================================================
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS Styles
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
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
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
# UTILITY FUNCTIONS
# ============================================================================
class LanguageUtils:
    """Utilities for language detection and handling."""

    # Arabic character range
    ARABIC_RANGE = ("\u0600", "\u06ff")

    # Arabic stop words for keyword extraction
    ARABIC_STOP_WORDS = {
        "من",
        "في",
        "على",
        "إلى",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "التي",
        "الذي",
        "هو",
        "هي",
        "هم",
        "هن",
        "أن",
        "إن",
        "كان",
        "كانت",
        "يكون",
        "تكون",
        "ما",
        "لا",
        "لم",
        "لن",
        "قد",
        "كل",
        "بعض",
        "أي",
        "و",
        "أو",
        "ثم",
        "حتى",
        "إذا",
        "لو",
        "كما",
        "بل",
        "لكن",
        "غير",
        "بين",
        "عند",
        "منذ",
        "حول",
        "خلال",
        "بعد",
        "قبل",
        "فوق",
        "تحت",
    }

    # English stop words
    ENGLISH_STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "from",
        "about",
        "into",
        "through",
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        if not text:
            return "en"

        arabic_chars = sum(
            1 for c in text if cls.ARABIC_RANGE[0] <= c <= cls.ARABIC_RANGE[1]
        )
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "en"

        return "ar" if (arabic_chars / total_alpha) > 0.3 else "en"

    @classmethod
    def get_stop_words(cls, lang: str) -> set:
        """Get stop words for the specified language."""
        return cls.ARABIC_STOP_WORDS if lang == "ar" else cls.ENGLISH_STOP_WORDS

    @classmethod
    def get_text_separators(cls, lang: str) -> List[str]:
        """Get text separators for chunking based on language."""
        if lang == "ar":
            return ["\n## ", "\n### ", "\n\n", "。", "\n", ". ", "، ", " ", ""]
        return ["\n## ", "\n### ", "\n\n", ". ", "\n", " ", ""]


class MathUtils:
    """Mathematical utility functions."""

    @staticmethod
    def sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)

    @staticmethod
    def cosine_sim(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity (assumes normalized vectors)."""
        return float(sum(x * y for x, y in zip(a, b)))


# ============================================================================
# LLM CLIENT
# ============================================================================
class LLMClient:
    """Wrapper for LLM API interactions."""

    def __init__(self, provider_key: str, api_key: str):
        config = LLM_PROVIDERS[provider_key]
        self.client = OpenAI(base_url=config["base_url"], api_key=api_key)
        self.model = config["model"]
        self.provider_name = config["name"]

    def chat(
        self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
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

    def translate_query(self, query: str, source_lang: str) -> Optional[str]:
        """Translate query to the other language for bilingual search."""
        if source_lang == "ar":
            instruction = (
                "Translate to English. Output ONLY the translation, nothing else:"
            )
        else:
            instruction = (
                "Translate to Arabic. Output ONLY the translation, nothing else:"
            )

        try:
            result = self.chat(
                [{"role": "user", "content": f"{instruction}\n{query}"}],
                temperature=0.3,
            )
            if (
                result
                and not result.startswith("Error:")
                and result.lower() != query.lower()
            ):
                return result.strip()
        except Exception:
            pass
        return None

    def generate_multi_queries(
        self, original_query: str, question_lang: str
    ) -> List[str]:
        """Generate multiple query variations for better retrieval coverage."""
        if question_lang == "ar":
            instruction = """أنشئ 3 صيغ بديلة لهذا السؤال بالعربية.
ركز على:
1. مصطلحات تقنية أكثر تحديداً
2. صيغة أوسع سياقاً
3. صياغة مختلفة بمرادفات

اكتب الأسئلة فقط، سؤال في كل سطر، بدون ترقيم."""
        else:
            instruction = """Generate 3 alternative ways to ask this question in English.
Focus on:
1. More specific technical terms
2. Broader context version
3. Different phrasing with synonyms

Output ONLY the questions, one per line, without numbering."""

        try:
            response = self.chat(
                [
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nOriginal Question: {original_query}",
                    }
                ],
                temperature=0.8,
            )

            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            cleaned = []
            for line in lines:
                line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)
                if line and len(line) > 10:
                    cleaned.append(line)

            return cleaned[: CONFIG.MULTI_QUERY_COUNT]
        except Exception as e:
            st.warning(f"Multi-query generation failed: {e}")
            return []


# ============================================================================
# MODEL LOADERS (Cached)
# ============================================================================
@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load the embedding model."""
    return SentenceTransformer(CONFIG.EMBEDDING_MODEL)


@st.cache_resource
def load_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker model."""
    return CrossEncoder(CONFIG.RERANK_MODEL)


def get_chroma_collection():
    """Get or create ChromaDB collection."""
    if "chroma_collection" not in st.session_state:
        chroma_client = PersistentClient(
            path=CONFIG.CHROMA_DIR, settings=Settings(allow_reset=True)
        )
        st.session_state["chroma_client"] = chroma_client
        st.session_state["chroma_collection"] = chroma_client.get_or_create_collection(
            "lecture_rag"
        )
    return st.session_state["chroma_client"], st.session_state["chroma_collection"]


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
class DocumentProcessor:
    """Handle document extraction and chunking."""

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def get_token_count(self, text: str) -> int:
        """Get actual token count using the embedding model's tokenizer."""
        try:
            tokenizer = self.embedder.tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception:
            return int(len(text.split()) * 1.3)

    def extract_text_from_pdf(self, file) -> List[Tuple[int, str]]:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file)
            pages = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((page_num, text))
            return pages
        except Exception as e:
            st.error(f"PDF error: {e}")
            return []

    def extract_text_from_ppt(self, file) -> List[Tuple[int, str]]:
        """Extract text from PowerPoint file."""
        try:
            prs = Presentation(file)
            slides = []
            for idx, slide in enumerate(prs.slides, start=1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                if slide_text:
                    slides.append((idx, "\n".join(slide_text)))
            return slides
        except Exception as e:
            st.error(f"PPT error: {e}")
            return []

    def split_text_recursive(
        self, text: str, max_size: int, overlap: int, separators: List[str]
    ) -> List[str]:
        """Recursively split text into chunks."""
        if self.get_token_count(text) <= max_size:
            return [text]

        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current_chunk = ""

                for i, part in enumerate(parts):
                    if sep and i > 0:
                        part = sep + part

                    test_chunk = current_chunk + part
                    if self.get_token_count(test_chunk) <= max_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        if overlap > 0 and chunks:
                            words = current_chunk.split()
                            overlap_text = " ".join(words[-overlap:])
                            current_chunk = overlap_text + part
                        else:
                            current_chunk = part

                if current_chunk:
                    chunks.append(current_chunk)

                if len(chunks) > 1:
                    return chunks

        # Force split by words
        words = text.split()
        chunks = []
        current = []
        current_tokens = 0

        for word in words:
            word_tokens = self.get_token_count(word)
            if current_tokens + word_tokens > max_size and current:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap > 0 else []
                current_tokens = sum(self.get_token_count(w) for w in current)
            current.append(word)
            current_tokens += word_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    def hierarchical_chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Create hierarchical parent-child chunks."""
        if not text or not text.strip():
            return []

        lang = LanguageUtils.detect_language(text)
        separators = LanguageUtils.get_text_separators(lang)

        # Create parent chunks
        parent_chunks = self.split_text_recursive(
            text, CONFIG.PARENT_CHUNK_SIZE, CONFIG.PARENT_OVERLAP, separators
        )

        all_chunks = []
        for parent_idx, parent_text in enumerate(parent_chunks):
            # Create child chunks from each parent
            child_chunks = self.split_text_recursive(
                parent_text, CONFIG.CHILD_CHUNK_SIZE, CONFIG.CHILD_OVERLAP, separators
            )

            for child_idx, child_text in enumerate(child_chunks):
                # Add context prefix for better embeddings
                context_prefix = f"""Document: {metadata['source']}
Page: {metadata['page']}
Type: {metadata['type']}
Language: {metadata['lang']}

"""
                child_with_context = context_prefix + child_text

                all_chunks.append(
                    {
                        "child_text": child_with_context,
                        "child_text_raw": child_text,
                        "parent_text": parent_text,
                        "parent_id": f"{metadata['source']}_p{metadata['page']}_parent{parent_idx}",
                        "metadata": {
                            **metadata,
                            "parent_idx": parent_idx,
                            "child_idx": child_idx,
                            "chunk_type": "child",
                        },
                    }
                )

        return all_chunks


class DocumentIngester:
    """Handle document ingestion into the vector database."""

    def __init__(self, embedder: SentenceTransformer, collection):
        self.embedder = embedder
        self.collection = collection
        self.processor = DocumentProcessor(embedder)

    @staticmethod
    def get_file_hash(file) -> str:
        """Generate MD5 hash of file contents."""
        file.seek(0)
        file_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        return file_hash

    def file_already_ingested(self, file_hash: str) -> bool:
        """Check if file has already been ingested."""
        try:
            results = self.collection.get(where={"file_hash": file_hash}, limit=1)
            return len(results["ids"]) > 0
        except Exception:
            return False

    def ingest(self, file) -> Tuple[int, str]:
        """Ingest a document into the vector database."""
        file_hash = self.get_file_hash(file)

        if self.file_already_ingested(file_hash):
            return 0, "exists"

        filename = file.name.lower()
        if filename.endswith(".pdf"):
            pages = self.processor.extract_text_from_pdf(file)
            source_type = "pdf"
        elif filename.endswith((".ppt", ".pptx")):
            pages = self.processor.extract_text_from_ppt(file)
            source_type = "ppt"
        else:
            return 0, "unsupported"

        if not pages:
            return 0, "no text"

        all_chunks = []
        chunk_metadata = []

        for page_num, text in pages:
            chunk_lang = LanguageUtils.detect_language(text)

            chunks = self.processor.hierarchical_chunk_text(
                text,
                {
                    "source": file.name,
                    "page": page_num,
                    "type": source_type,
                    "file_hash": file_hash,
                    "lang": chunk_lang,
                },
            )

            for chunk_data in chunks:
                all_chunks.append(f"passage: {chunk_data['child_text']}")
                chunk_metadata.append(
                    {
                        "child_text": chunk_data["child_text"],
                        "child_text_raw": chunk_data["child_text_raw"],
                        "parent_text": chunk_data["parent_text"],
                        "parent_id": chunk_data["parent_id"],
                        **chunk_data["metadata"],
                    }
                )

        if not all_chunks:
            return 0, "no chunks"

        try:
            embeddings = self.embedder.encode(
                all_chunks, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            self.collection.add(
                ids=[str(uuid.uuid4()) for _ in all_chunks],
                documents=[m["child_text_raw"] for m in chunk_metadata],
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": m["source"],
                        "page": m["page"],
                        "type": m["type"],
                        "file_hash": m["file_hash"],
                        "lang": m["lang"],
                        "parent_text": m["parent_text"],
                        "parent_id": m["parent_id"],
                        "parent_idx": m.get("parent_idx", 0),
                        "child_idx": m.get("child_idx", 0),
                        "chunk_type": m.get("chunk_type", "child"),
                    }
                    for m in chunk_metadata
                ],
            )
            return len(all_chunks), "success"
        except Exception as e:
            return 0, str(e)


# ============================================================================
# RETRIEVAL SYSTEM
# ============================================================================
class Retriever:
    """Handle document retrieval with hybrid search and reranking."""

    def __init__(
        self, embedder: SentenceTransformer, collection, llm_client: LLMClient
    ):
        self.embedder = embedder
        self.collection = collection
        self.llm_client = llm_client
        self.reranker = None

        if CONFIG.ENABLE_RERANKING or CONFIG.ENABLE_CRAG:
            try:
                self.reranker = load_reranker()
            except Exception as e:
                st.warning(f"Reranker failed to load: {e}")

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text."""
        lang = LanguageUtils.detect_language(text)
        stop_words = LanguageUtils.get_stop_words(lang)

        # Tokenize based on language
        if lang == "ar":
            words = re.findall(r"[\u0600-\u06ff]+|\b\w+\b", text.lower())
        else:
            words = re.findall(r"\b\w+\b", text.lower())

        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Return unique keywords
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
            if len(unique_keywords) >= max_keywords:
                break

        return unique_keywords

    def reciprocal_rank_fusion(self, results_list: List, k: int = 60) -> List[Dict]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        doc_data = {}

        for results in results_list:
            if not results or "ids" not in results or not results["ids"]:
                continue

            for rank, (doc_id, distance, metadata, document) in enumerate(
                zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                    results["documents"][0],
                )
            ):
                scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_data:
                    doc_data[doc_id] = {
                        "metadata": metadata,
                        "document": document,
                        "distance": distance,
                    }

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {"id": doc_id, "score": score, **doc_data[doc_id]}
            for doc_id, score in sorted_docs
        ]

    def hybrid_retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """Perform hybrid search combining vector and keyword search."""
        question_embedding = self.embedder.encode(
            [f"query: {question}"], normalize_embeddings=True
        ).tolist()

        # Vector search
        vector_results = self.collection.query(
            query_embeddings=question_embedding, n_results=top_k * 3
        )

        # Keyword search
        keywords = self.extract_keywords(question)
        keyword_results = None

        if keywords:
            try:
                keyword_query = " ".join(keywords)
                keyword_results = self.collection.query(
                    query_embeddings=question_embedding,
                    n_results=top_k * 3,
                    where_document={"$contains": keyword_query},
                )
            except Exception:
                keyword_results = None

        # Fusion
        if keyword_results and keyword_results["ids"] and keyword_results["ids"][0]:
            fused = self.reciprocal_rank_fusion([vector_results, keyword_results])
        else:
            fused = [
                {
                    "id": doc_id,
                    "score": 1.0 / (1 + dist),
                    "metadata": meta,
                    "document": doc,
                    "distance": dist,
                }
                for doc_id, dist, meta, doc in zip(
                    vector_results["ids"][0],
                    vector_results["distances"][0],
                    vector_results["metadatas"][0],
                    vector_results["documents"][0],
                )
            ]

        return fused[:top_k]

    def dedupe_by_parent(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results by parent_id."""
        seen_parent_ids = set()
        deduped = []
        for r in results:
            parent_id = r["metadata"].get("parent_id", r["id"])
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
            deduped.append(r)
        return deduped

    def truncate_for_rerank(self, text: str, max_tokens: int = 256) -> str:
        """Truncate text for reranking to avoid tokenizer overflow."""
        if not text or not self.reranker:
            return text[:2000] if text else ""

        try:
            tok = self.reranker.tokenizer
            ids = tok.encode(
                text, add_special_tokens=True, truncation=True, max_length=max_tokens
            )
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            return text[:2000]

    def rerank(
        self, query: str, candidates: List[Dict], top_n: int
    ) -> Tuple[List[Dict], List[float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates or not self.reranker:
            return candidates[:top_n], []

        pool = candidates[: min(len(candidates), CONFIG.RERANK_CANDIDATES)]

        pairs = []
        for c in pool:
            doc_text = c.get("parent_text") or c.get("child_text", "")
            doc_text = self.truncate_for_rerank(doc_text)
            pairs.append([query, doc_text])

        raw_scores = self.reranker.predict(pairs)
        raw_scores = [float(s) for s in raw_scores]

        for c, s in zip(pool, raw_scores):
            c["rerank_score"] = s
            c["rerank_prob"] = MathUtils.sigmoid(s)

        pool.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        reranked = pool[: min(top_n, len(pool))]
        reranked_scores = [float(x.get("rerank_score", 0.0)) for x in reranked]

        return reranked, reranked_scores

    def assess_retrieval_quality(self, rerank_scores: List[float]) -> Dict[str, float]:
        """Assess retrieval quality for CRAG gating."""
        if not rerank_scores:
            return {"max_relevance": 0.0, "mean_top3": 0.0}

        probs = [MathUtils.sigmoid(s) for s in rerank_scores]
        probs_sorted = sorted(probs, reverse=True)

        max_rel = probs_sorted[0]
        top3 = probs_sorted[: min(3, len(probs_sorted))]
        mean_top3 = sum(top3) / len(top3)

        return {"max_relevance": float(max_rel), "mean_top3": float(mean_top3)}

    def should_correct(self, metrics: Dict[str, float]) -> bool:
        """Determine if CRAG corrective retrieval is needed."""
        if metrics["max_relevance"] < CONFIG.CRAG_MIN_MAX_RELEVANCE:
            return True
        if metrics["mean_top3"] < CONFIG.CRAG_MIN_MEAN_TOP3:
            return True
        return False

    def retrieve(
        self, question: str, per_query_k: int = 3, final_k: int = None
    ) -> Tuple[List[Dict], Dict]:
        """Full retrieval pipeline with multi-query, hybrid search, reranking, and CRAG."""
        if final_k is None:
            final_k = CONFIG.TOP_K

        debug = {
            "crag_used": False,
            "crag_metrics": None,
            "crag_metrics_after": None,
        }

        question_lang = LanguageUtils.detect_language(question)

        # Generate query variations
        query_variations = [question]
        query_variations.extend(
            self.llm_client.generate_multi_queries(question, question_lang)
        )

        # Add translated query
        translated = self.llm_client.translate_query(question, question_lang)
        if translated:
            query_variations.append(translated)

        # Retrieve for each query variation
        all_results = []
        for q in query_variations:
            results = self.hybrid_retrieve(q, top_k=per_query_k)
            for result in results:
                all_results.append(
                    {
                        "id": result["id"],
                        "score": float(result["score"]),
                        "child_text": result["document"],
                        "parent_text": result["metadata"].get(
                            "parent_text", result["document"]
                        ),
                        "metadata": result["metadata"],
                    }
                )

        # Sort and dedupe
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = self.dedupe_by_parent(all_results)
        candidates = all_results[: max(final_k, CONFIG.RERANK_CANDIDATES)]

        # Rerank
        reranked = candidates
        rerank_scores = []

        if CONFIG.ENABLE_RERANKING and self.reranker and candidates:
            reranked, rerank_scores = self.rerank(
                question, candidates, CONFIG.RERANK_TOP_K
            )
        else:
            reranked = candidates[:final_k]

        # CRAG - Corrective RAG
        if CONFIG.ENABLE_CRAG and self.reranker:
            if not rerank_scores and reranked:
                _, rerank_scores = self.rerank(
                    question, reranked, min(len(reranked), final_k)
                )

            metrics = self.assess_retrieval_quality(rerank_scores)
            debug["crag_metrics"] = metrics

            if self.should_correct(metrics):
                debug["crag_used"] = True

                # Expanded retrieval
                expanded_results = []
                for q in query_variations:
                    results = self.hybrid_retrieve(
                        q, top_k=per_query_k * CONFIG.CRAG_RETRIEVE_EXPANSION
                    )
                    for result in results:
                        expanded_results.append(
                            {
                                "id": result["id"],
                                "score": float(result["score"]),
                                "child_text": result["document"],
                                "parent_text": result["metadata"].get(
                                    "parent_text", result["document"]
                                ),
                                "metadata": result["metadata"],
                            }
                        )

                expanded_results.sort(key=lambda x: x["score"], reverse=True)
                expanded_results = self.dedupe_by_parent(expanded_results)

                if expanded_results:
                    corrected, corrected_scores = self.rerank(
                        question, expanded_results, CONFIG.RERANK_TOP_K
                    )
                    corrected_metrics = self.assess_retrieval_quality(corrected_scores)
                    debug["crag_metrics_after"] = corrected_metrics

                    if (
                        corrected_metrics["max_relevance"] > metrics["max_relevance"]
                    ) or (corrected_metrics["mean_top3"] > metrics["mean_top3"]):
                        reranked = corrected

                    if (
                        self.should_correct(corrected_metrics)
                        and corrected_metrics["max_relevance"] < 0.20
                    ):
                        reranked = []

        return reranked, debug


# ============================================================================
# SEMANTIC CACHE
# ============================================================================
class SemanticCache:
    """Semantic caching for query results."""

    def __init__(self, cache: OrderedDict, embedder: SentenceTransformer):
        self.cache = cache
        self.embedder = embedder

    @staticmethod
    def make_scope_key(
        provider_key: str, model_name: str, collection_count: int
    ) -> str:
        """Create a scope key for cache isolation."""
        return f"{provider_key}::{model_name}::count:{collection_count}"

    def lookup(
        self, query: str, query_embedding: List[float], scope_key: str
    ) -> Optional[Dict]:
        """Look up a semantically similar cached entry."""
        if not CONFIG.ENABLE_SEMANTIC_CACHE:
            return None

        best_key = None
        best_sim = -1.0
        best_payload = None

        for k, payload in self.cache.items():
            if not k.startswith(scope_key + "::"):
                continue

            emb = payload.get("query_embedding")
            if not emb:
                continue

            sim = MathUtils.cosine_sim(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_key = k
                best_payload = payload

        if best_payload and best_sim >= CONFIG.CACHE_SIM_THRESHOLD:
            self.cache.move_to_end(best_key, last=True)
            result = dict(best_payload)
            result["cache_similarity"] = float(best_sim)
            return result

        return None

    def store(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        sources: List[Dict],
        followups: List[str],
        question_lang: str,
        scope_key: str,
    ):
        """Store a query result in cache."""
        if not CONFIG.ENABLE_SEMANTIC_CACHE:
            return

        qh = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]
        k = f"{scope_key}::{qh}"

        self.cache[k] = {
            "query": query,
            "query_embedding": query_embedding,
            "answer": answer,
            "sources": sources,
            "followups": followups,
            "question_lang": question_lang,
            "cached_at": time.time(),
        }
        self.cache.move_to_end(k, last=True)

        # LRU eviction
        while len(self.cache) > CONFIG.CACHE_MAX_ENTRIES:
            self.cache.popitem(last=False)


# ============================================================================
# ANSWER GENERATION
# ============================================================================
class AnswerGenerator:
    """Generate answers with proper citations and language handling."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def validate_arabic_response(self, text: str) -> str:
        """Validate and clean Arabic responses."""
        # Remove Chinese characters
        cleaned = re.sub(r"[\u4e00-\u9fff]", "", text)

        # Remove Vietnamese diacritics
        vietnamese_pattern = (
            r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]"
        )
        cleaned = re.sub(vietnamese_pattern, "", cleaned, flags=re.IGNORECASE)

        # Check for excessive Latin text outside parentheses
        text_outside_parens = re.sub(r"\([^)]*\)", "", cleaned)
        latin_outside = re.findall(
            r"(?<!\()[a-zA-Z]{3,}(?![^(]*\))", text_outside_parens
        )

        if latin_outside and len(latin_outside) > 3:
            unique_terms = list(set(latin_outside))[:5]
            st.warning(
                f"⚠️ Language mixing detected. Terms found: {', '.join(unique_terms)}"
            )

        return cleaned

    def get_arabic_system_prompt(self) -> str:
        """Get the system prompt for Arabic responses."""
        return """أنت مدرس جامعي متخصص في مجالات هندسة الذكاء الاصطناعي (الرؤية الحاسوبية، معالجة اللغات الطبيعية، التعلم الآلي).

═══════════════════════════════════════════════════════════════
قواعد اللغة - إلزامية
═══════════════════════════════════════════════════════════════
- اكتب بالعربية الفصحى فقط
- المصطلحات التقنية: اكتب بالعربية أولاً ثم (English) بين قوسين
- مثال صحيح: "الرؤية الحاسوبية (Computer Vision)"
- مثال خاطئ: "Computer Vision هو مجال" أو "الـ processing"

═══════════════════════════════════════════════════════════════
قواعد الاستشهاد - إلزامية
═══════════════════════════════════════════════════════════════
- أضف [رقم] بعد كل معلومة مباشرة
- مثال: "الرؤية الحاسوبية مجال متعدد التخصصات [1]."
- لا تذكر معلومات بدون استشهاد
- إذا لم تجد المعلومة في المصادر، قل: "لا تتوفر هذه المعلومة في المواد المتاحة"

═══════════════════════════════════════════════════════════════
تنسيق الإجابة
═══════════════════════════════════════════════════════════════
- اكتب فقرات متصلة (ليس نقاط)
- استخدم علامات الترقيم العربية: ، ؛ ؟
- أسلوب أكاديمي واضح
- 150-250 كلمة تقريباً"""

    def get_english_system_prompt(self) -> str:
        """Get the system prompt for English responses."""
        return """You are a university tutor specializing in AI Engineering (Computer Vision, NLP, Machine Learning).

═══════════════════════════════════════════════════════════════
CITATION REQUIREMENTS - MANDATORY
═══════════════════════════════════════════════════════════════
- Add [number] IMMEDIATELY after every factual statement
- Example: "Computer vision is an interdisciplinary field [1]."
- Never make claims without citations
- If information not found, say: "This information is not available in the provided materials"

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════
- Write in clear, flowing paragraphs (NOT bullet points)
- Academic but accessible language
- Translate any Arabic content from sources to English
- Aim for 150-250 words"""

    def generate_answer(
        self, retrieved_docs: List[Dict], question: str
    ) -> Tuple[str, List[Dict]]:
        """Generate an answer with citations."""
        question_lang = LanguageUtils.detect_language(question)

        if not retrieved_docs:
            no_info = (
                "لا تتوفر معلومات ذات صلة في المستندات المرفوعة."
                if question_lang == "ar"
                else "I couldn't find relevant information in the uploaded documents."
            )
            return no_info, []

        # Build context from parent texts
        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs, 1):
            parent_text = doc.get("parent_text", doc.get("child_text", ""))
            context_parts.append(f"[Source {i}]:\n{parent_text}")
            sources.append(doc["metadata"])

        context = "\n\n".join(context_parts)

        # Select system prompt based on language
        if question_lang == "ar":
            system_prompt = self.get_arabic_system_prompt()
            reminder = "تذكير: اكتب بالعربية فقط، مع المصطلحات التقنية بين قوسين بالإنجليزية. استشهد بكل معلومة."
        else:
            system_prompt = self.get_english_system_prompt()
            reminder = "REMINDER: Write in English only. Cite every fact."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""LECTURE MATERIAL SOURCES:
{context}

STUDENT QUESTION: {question}

{reminder}""",
            },
        ]

        # Lower temperature for Arabic to reduce hallucination
        temperature = 0.3 if question_lang == "ar" else 0.5

        answer = self.llm_client.chat(messages, temperature=temperature)

        # Validate Arabic responses
        if question_lang == "ar":
            answer = self.validate_arabic_response(answer)

            # Check citation count
            citations = re.findall(r"\[\d+\]", answer)
            if len(citations) < 2:
                st.warning(
                    "⚠️ Response has few citations - answer quality may be limited"
                )

        return answer, sources

    def generate_followups(
        self, question: str, answer: str, question_lang: str
    ) -> List[str]:
        """Generate follow-up questions."""
        if question_lang == "ar":
            instruction = """بناءً على هذا السؤال والجواب، أنشئ 3 أسئلة متابعة قصيرة بالعربية.
ركز على: توضيح المفاهيم، مواضيع ذات صلة، أمثلة تطبيقية.
اكتب الأسئلة فقط، سؤال في كل سطر، بدون ترقيم."""
        else:
            instruction = """Based on this Q&A, generate 3 short follow-up questions in English.
Focus on: clarifying concepts, related topics, practical examples.
Output ONLY the questions, one per line, without numbering."""

        try:
            result = self.llm_client.chat(
                [
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nQuestion: {question}\n\nAnswer: {answer[:500]}...",
                    }
                ],
                temperature=0.8,
            )

            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            cleaned = []
            for line in lines[:3]:
                line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)
                if line and len(line) > 10:
                    cleaned.append(line)
            return cleaned
        except Exception:
            return []


# ============================================================================
# RAG PIPELINE
# ============================================================================
class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""

    def __init__(
        self, llm_client: LLMClient, embedder: SentenceTransformer, collection
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.collection = collection
        self.retriever = Retriever(embedder, collection, llm_client)
        self.answer_generator = AnswerGenerator(llm_client)
        self.cache = SemanticCache(st.session_state.semantic_cache, embedder)

    def answer(self, question: str) -> Tuple[str, List[Dict], List[str], Dict]:
        """Process a question and return answer, sources, followups, and debug info."""
        question_lang = LanguageUtils.detect_language(question)
        debug = {
            "cache_hit": False,
            "cache_similarity": None,
            "crag_used": False,
            "crag_metrics": None,
        }

        # Get query embedding
        query_emb = self.embedder.encode(
            [f"query: {question}"], normalize_embeddings=True
        ).tolist()[0]

        # Create cache scope key
        try:
            collection_count = self.collection.count()
        except Exception:
            collection_count = 0

        scope_key = SemanticCache.make_scope_key(
            st.session_state.selected_provider, self.llm_client.model, collection_count
        )

        # Check cache
        if CONFIG.ENABLE_SEMANTIC_CACHE:
            hit = self.cache.lookup(question, query_emb, scope_key)
            if hit:
                debug["cache_hit"] = True
                debug["cache_similarity"] = hit.get("cache_similarity")
                return (
                    hit["answer"],
                    hit.get("sources", []),
                    hit.get("followups", []),
                    debug,
                )

        # Retrieve documents
        retrieved_docs, retrieval_debug = self.retriever.retrieve(question)
        debug.update(retrieval_debug)

        # Generate answer
        answer, sources = self.answer_generator.generate_answer(
            retrieved_docs, question
        )

        # Generate follow-ups
        followups = self.answer_generator.generate_followups(
            question, answer, question_lang
        )

        # Store in cache
        if CONFIG.ENABLE_SEMANTIC_CACHE:
            self.cache.store(
                query=question,
                query_embedding=query_emb,
                answer=answer,
                sources=sources,
                followups=followups,
                question_lang=question_lang,
                scope_key=scope_key,
            )

        return answer, sources, followups, debug


# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIRenderer:
    """Render UI components."""

    @staticmethod
    def render_source_cards(metadatas: List[Dict]):
        """Render source cards."""
        if not metadatas:
            return

        st.markdown(
            '<div class="section-label">📚 Sources</div>', unsafe_allow_html=True
        )

        cols = st.columns(min(len(metadatas), 3))

        for i, meta in enumerate(metadatas):
            col_idx = i % 3
            with cols[col_idx]:
                source_name = meta.get("source", "Unknown")
                page = meta.get("page", "?")
                lang = meta.get("lang", "en")

                display_name = (
                    source_name[:25] + "..." if len(source_name) > 25 else source_name
                )
                page_label = "صفحة" if lang == "ar" else "Page"

                st.markdown(
                    f"""
                <div class="source-card">
                    <div class="source-number">{i + 1}</div>
                    <div style="overflow: hidden;">
                        <div style="font-weight: 500; font-size: 0.85rem; color: #1a1a1a;
                                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            {display_name}
                        </div>
                        <div style="color: #666; font-size: 0.75rem;">{page_label} {page}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    @staticmethod
    def render_answer(answer: str, is_rtl: bool = False):
        """Render the answer with styled citations."""

        def replace_citation(match):
            return f'<span class="citation">{match.group(1)}</span>'

        styled = re.sub(r"\[(\d+)\]", replace_citation, answer)
        styled = styled.replace("\n\n", "</p><p>").replace("\n", "<br>")
        styled = f"<p>{styled}</p>"

        css_class = "assistant-message-rtl" if is_rtl else "assistant-message"
        st.markdown(f'<div class="{css_class}">{styled}</div>', unsafe_allow_html=True)

    @staticmethod
    def render_followups(questions: List[str], question_lang: str) -> Optional[str]:
        """Render follow-up question buttons."""
        if not questions:
            return None

        label = "أسئلة ذات صلة" if question_lang == "ar" else "Related Questions"
        st.markdown(
            f'<div class="section-label">💡 {label}</div>', unsafe_allow_html=True
        )

        cols = st.columns(len(questions))
        for i, (col, q) in enumerate(zip(cols, questions)):
            with col:
                if st.button(
                    q, key=f"followup_{i}_{hash(q)}", use_container_width=True
                ):
                    return q
        return None

    @classmethod
    def render_message(
        cls,
        role: str,
        content: str,
        sources: List[Dict] = None,
        followups: List[str] = None,
        question_lang: str = "en",
    ) -> Optional[str]:
        """Render a chat message."""
        if role == "user":
            is_rtl = LanguageUtils.detect_language(content) == "ar"
            direction = 'dir="rtl"' if is_rtl else ""
            st.markdown(
                f'<div class="user-message" {direction}>{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            if sources:
                cls.render_source_cards(sources)
            cls.render_answer(content, LanguageUtils.detect_language(content) == "ar")
            if followups:
                return cls.render_followups(followups, question_lang)
        return None


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    """Render the sidebar with settings and document upload."""
    with st.sidebar:
        st.markdown("### ⚙️ LLM Provider")

        selected = st.selectbox(
            "Choose provider",
            options=list(LLM_PROVIDERS.keys()),
            index=list(LLM_PROVIDERS.keys()).index(st.session_state.selected_provider),
            label_visibility="collapsed",
        )
        st.session_state.selected_provider = selected

        provider_config = LLM_PROVIDERS[selected]
        st.caption(f"ℹ️ {provider_config['notes']}")

        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder=f"Enter {provider_config['name']} API key",
        )
        if api_key:
            st.session_state.api_key = api_key

        st.markdown(f"[🔑 Get free API key]({provider_config['get_key_url']})")

        st.markdown("---")
        st.markdown("### 📄 Documents")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "ppt", "pptx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("📥 Ingest", use_container_width=True):
            if not uploaded_files:
                st.error("Upload files first")
            else:
                embedder = load_embedder()
                _, collection = get_chroma_collection()
                ingester = DocumentIngester(embedder, collection)

                total = 0
                for file in uploaded_files:
                    with st.spinner(f"Processing {file.name}..."):
                        added, status = ingester.ingest(file)
                        if status == "exists":
                            st.info(f"✓ {file.name} already indexed")
                        elif status == "success":
                            st.success(f"✓ {file.name}: {added} chunks added")
                            total += added
                        else:
                            st.error(f"✗ {file.name}: {status}")

                if total > 0:
                    st.success(f"✅ Total: {total} new chunks indexed")

        # Show document count
        try:
            _, collection = get_chroma_collection()
            count = collection.count()
            st.caption(f"📊 {count} chunks in database")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### 🔧 Advanced Settings")

        with st.expander("Retrieval Settings"):
            st.caption("Currently using:")
            st.caption(f"• Parent chunks: {CONFIG.PARENT_CHUNK_SIZE} tokens")
            st.caption(f"• Child chunks: {CONFIG.CHILD_CHUNK_SIZE} tokens")
            st.caption(f"• Top-K results: {CONFIG.TOP_K}")
            st.caption(f"• Multi-query count: {CONFIG.MULTI_QUERY_COUNT}")
            st.caption("• Hybrid search: Vector + Keyword")
            st.caption("• Parent-child chunking: Enabled")
            st.caption(
                f"• Reranking: {'Enabled' if CONFIG.ENABLE_RERANKING else 'Disabled'}"
            )
            st.caption(
                f"• Semantic cache: {'Enabled' if CONFIG.ENABLE_SEMANTIC_CACHE else 'Disabled'}"
            )
            st.caption(f"• CRAG: {'Enabled' if CONFIG.ENABLE_CRAG else 'Disabled'}")

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
                    st.success("Database reset!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        if CONFIG.ENABLE_SEMANTIC_CACHE:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.semantic_cache = OrderedDict()
                st.success("Cache cleared!")
                st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    render_sidebar()

    # Welcome header when no chat history
    if not st.session_state.chat_history:
        st.markdown(
            """
        <div class="app-header">
            <div class="app-title">🎓 University Tutor - AI Engineering Specialist</div>
            <div class="app-subtitle">Ask questions about your lecture materials in English or Arabic</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        provider_name = LLM_PROVIDERS[st.session_state.selected_provider]["name"]
        model_name = LLM_PROVIDERS[st.session_state.selected_provider]["model"]
        st.markdown(
            f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="provider-badge">⚡ Powered by {provider_name} • {model_name.split('/')[-1]}</span>
            <br><br>
            <span class="provider-badge">🚀 Enhanced RAG: Hierarchical Chunking • Hybrid Search • 
            Multi-Query • Reranking • CRAG • Semantic Cache</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Check for API key
    if not st.session_state.api_key:
        st.info(
            "👈 Select a provider and enter your API key in the sidebar to get started"
        )
        st.stop()

    # Initialize LLM client
    llm_client = LLMClient(st.session_state.selected_provider, st.session_state.api_key)

    # Render chat history
    followup_clicked = None
    for msg in st.session_state.chat_history:
        result = UIRenderer.render_message(
            msg["role"],
            msg["content"],
            msg.get("sources"),
            msg.get("followups"),
            msg.get("question_lang", "en"),
        )
        if result:
            followup_clicked = result

    # Handle followup clicks
    if followup_clicked:
        st.session_state.pending_question = followup_clicked
        st.rerun()

    # Chat input
    question = st.chat_input(
        "Ask anything about your lecture materials... / اسأل أي سؤال عن محاضراتك..."
    )

    # Handle pending questions from followups
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    # Process question
    if question:
        question = question.strip()
        if question:
            question_lang = LanguageUtils.detect_language(question)
            st.session_state.chat_history.append({"role": "user", "content": question})
            UIRenderer.render_message("user", question)

            with st.spinner("🔍 Searching lecture materials and analyzing..."):
                embedder = load_embedder()
                _, collection = get_chroma_collection()

                pipeline = RAGPipeline(llm_client, embedder, collection)
                answer, sources, followups, debug = pipeline.answer(question)

                # Show debug info
                if debug.get("cache_hit"):
                    sim = debug.get("cache_similarity")
                    if sim is not None:
                        st.caption(f"Cache hit (similarity: {sim:.3f})")

                if debug.get("crag_used") and debug.get("crag_metrics"):
                    m = debug["crag_metrics"]
                    st.caption(
                        f"CRAG corrective retrieval used (max_rel={m['max_relevance']:.2f})"
                    )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "followups": followups,
                    "question_lang": question_lang,
                }
            )
            st.rerun()

    # Footer
    st.markdown("---")
    provider_name = LLM_PROVIDERS[st.session_state.selected_provider]["name"]
    st.markdown(
        f"""
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <strong>Enhanced RAG Architecture:</strong> Hierarchical Parent-Child Chunking • 
        Hybrid Search (Vector + Keyword) • Multi-Query Generation • Cross-lingual Retrieval • 
        Cross-Encoder Reranking • CRAG Validation • Semantic Cache<br>
        <strong>Powered by:</strong> {provider_name} • ChromaDB • E5-multilingual-small
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
