# app.py
# Enhanced Multilingual RAG Chat - University Tutor
# Improvements: Hierarchical chunking, Hybrid search, Multi-query generation, Enhanced Arabic support

import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
from pptx import Presentation
from pypdf import PdfReader
from openai import OpenAI
import hashlib
import uuid
import re
from collections import defaultdict
from typing import List, Dict, Tuple


# ================================
# CONFIGURATION
# ================================
PARENT_CHUNK_SIZE = 1500
PARENT_OVERLAP = 200
CHILD_CHUNK_SIZE = 512
CHILD_OVERLAP = 50
TOP_K = 5
MULTI_QUERY_COUNT = 3

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# LLM Provider configurations
LLM_PROVIDERS = {
    "Groq (Llama 3.3 70B) ⚡ Recommended": {
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

# ================================
# PAGE SETUP & STYLING
# ================================
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Perplexity-inspired CSS
st.markdown(
    """
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
    
    .sources-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin: 1rem 0;
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
    }
    
    .source-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102,126,234,0.15);
    }
    
    .source-number {
        background: #667eea;
        color: white;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .source-info { overflow: hidden; }
    
    .source-title {
        font-weight: 500;
        color: #1a1a1a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .source-meta {
        color: #666;
        font-size: 0.75rem;
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
""",
    unsafe_allow_html=True,
)


# ================================
# SESSION STATE
# ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = list(LLM_PROVIDERS.keys())[0]


# ================================
# LLM CLIENT
# ================================
def get_llm_client(provider_key, api_key):
    """Create OpenAI-compatible client for the selected provider."""
    config = LLM_PROVIDERS[provider_key]
    return OpenAI(base_url=config["base_url"], api_key=api_key), config["model"]


def chat_completion(client, model, messages, temperature=0.7):
    """Generate chat completion using the provider."""
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ================================
# LOAD MODELS & DATABASE
# ================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_chroma_collection():
    if "chroma_collection" not in st.session_state:
        chroma_client = PersistentClient(
            path=CHROMA_DIR, settings=Settings(allow_reset=True)
        )
        st.session_state["chroma_client"] = chroma_client
        st.session_state["chroma_collection"] = chroma_client.get_or_create_collection(
            "lecture_rag"
        )
    return st.session_state["chroma_client"], st.session_state["chroma_collection"]


# ================================
# LANGUAGE UTILITIES
# ================================
def detect_language(text):
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"
    return "ar" if (arabic_chars / total_alpha) > 0.3 else "en"


def translate_query(client, model, query, source_lang):
    """Translate query to the other language for bilingual search."""
    if source_lang == "ar":
        instruction = "Translate to English. Output ONLY the translation, nothing else:"
    else:
        instruction = "Translate to Arabic. Output ONLY the translation, nothing else:"

    try:
        result = chat_completion(
            client,
            model,
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


# ================================
# ENHANCED CHUNKING
# ================================
def get_token_count(embedder, text):
    """Get actual token count using the embedding model's tokenizer."""
    try:
        # Access the tokenizer from the sentence transformer
        tokenizer = embedder.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception:
        # Fallback: approximate using whitespace (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.3)


def hierarchical_chunk_text(
    text,
    metadata,
    embedder,
    parent_size=PARENT_CHUNK_SIZE,
    parent_overlap=PARENT_OVERLAP,
    child_size=CHILD_CHUNK_SIZE,
    child_overlap=CHILD_OVERLAP,
):
    """
    Create parent-child chunks for hierarchical retrieval.
    Returns chunks with both child (for retrieval) and parent (for context) text.
    """
    if not text or not text.strip():
        return []

    # Detect language for appropriate separators
    lang = detect_language(text)

    # Language-specific separators
    if lang == "ar":
        separators = [
            "\n## ",
            "\n### ",  # Headers
            "\n\n",  # Paragraphs
            "。",  # Arabic period
            "\n",  # Lines
            ". ",  # Period
            " ",
            "",
        ]
    else:
        separators = [
            "\n## ",
            "\n### ",  # Headers
            "\n\n",  # Paragraphs
            ". ",  # Sentences
            "\n",  # Lines
            " ",
            "",
        ]

    # Recursive split function
    def split_text_recursive(text, max_size, overlap, separators):
        """Recursively split text using hierarchical separators."""
        if get_token_count(embedder, text) <= max_size:
            return [text]

        # Try each separator in order
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""

                for i, part in enumerate(parts):
                    # Add separator back (except for empty string separator)
                    if separator and i > 0:
                        part = separator + part

                    # Check if adding this part would exceed size
                    test_chunk = current_chunk + part
                    if get_token_count(embedder, test_chunk) <= max_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is full, save it
                        if current_chunk:
                            chunks.append(current_chunk)

                        # Start new chunk with overlap
                        if overlap > 0 and chunks:
                            # Get last N tokens for overlap
                            words = current_chunk.split()
                            overlap_text = " ".join(words[-overlap:])
                            current_chunk = overlap_text + part
                        else:
                            current_chunk = part

                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # If we successfully split, return
                if len(chunks) > 1:
                    return chunks

        # If no separator worked, force split by tokens
        words = text.split()
        chunks = []
        current = []
        current_tokens = 0

        for word in words:
            word_tokens = get_token_count(embedder, word)
            if current_tokens + word_tokens > max_size and current:
                chunks.append(" ".join(current))
                # Overlap
                current = current[-overlap:] if overlap > 0 else []
                current_tokens = sum(get_token_count(embedder, w) for w in current)
            current.append(word)
            current_tokens += word_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    # Create parent chunks
    parent_chunks = split_text_recursive(text, parent_size, parent_overlap, separators)

    # Create child chunks from each parent
    all_chunks = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        child_chunks = split_text_recursive(
            parent_text, child_size, child_overlap, separators
        )

        for child_idx, child_text in enumerate(child_chunks):
            # Add document context to child
            context_prefix = f"""Document: {metadata['source']}
Page: {metadata['page']}
Type: {metadata['type']}
Language: {metadata['lang']}

"""
            child_with_context = context_prefix + child_text

            all_chunks.append(
                {
                    "child_text": child_with_context,
                    "child_text_raw": child_text,  # Without context prefix
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


# ================================
# DOCUMENT PROCESSING
# ================================
def extract_text_from_pdf(file):
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


def extract_text_from_ppt(file):
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


def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


def file_already_ingested(collection, file_hash):
    try:
        results = collection.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def ingest_document(file, embedder, collection):
    file_hash = get_file_hash(file)

    if file_already_ingested(collection, file_hash):
        return 0, "exists"

    filename = file.name.lower()
    if filename.endswith(".pdf"):
        pages = extract_text_from_pdf(file)
        source_type = "pdf"
    elif filename.endswith((".ppt", ".pptx")):
        pages = extract_text_from_ppt(file)
        source_type = "ppt"
    else:
        return 0, "unsupported"

    if not pages:
        return 0, "no text"

    all_chunks = []
    chunk_metadata = []

    for page_num, text in pages:
        # Detect language
        chunk_lang = detect_language(text)

        # Create hierarchical chunks
        chunks = hierarchical_chunk_text(
            text,
            {
                "source": file.name,
                "page": page_num,
                "type": source_type,
                "file_hash": file_hash,
                "lang": chunk_lang,
            },
            embedder,
        )

        for chunk_data in chunks:
            # Embed the child text (with context prefix for better embeddings)
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
        embeddings = embedder.encode(
            all_chunks, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        collection.add(
            ids=[str(uuid.uuid4()) for _ in all_chunks],
            documents=[
                m["child_text_raw"] for m in chunk_metadata
            ],  # Store raw for display
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


# ================================
# ENHANCED RETRIEVAL
# ================================
def extract_keywords(text, max_keywords=5):
    """Simple keyword extraction (can be enhanced with spaCy)."""
    # Remove common words and keep technical terms
    stop_words = {
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
    }

    # Tokenize and clean
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


def reciprocal_rank_fusion(results_list, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    Formula: score(d) = Σ 1/(k + rank_i(d))
    """
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

    # Sort by RRF score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"id": doc_id, "score": score, **doc_data[doc_id]}
        for doc_id, score in sorted_docs
    ]


def hybrid_retrieve(question, collection, embedder, top_k=5):
    """
    Hybrid search: Combine vector search + keyword filtering.
    """
    # Vector search
    question_embedding = embedder.encode(
        [f"query: {question}"], normalize_embeddings=True
    ).tolist()

    vector_results = collection.query(
        query_embeddings=question_embedding, n_results=top_k * 3  # Get more for fusion
    )

    # Keyword search (metadata filtering for exact term matches)
    keywords = extract_keywords(question)

    keyword_results = None
    if keywords:
        try:
            # Build keyword query
            keyword_query = " ".join(keywords)
            keyword_results = collection.query(
                query_embeddings=question_embedding,
                n_results=top_k * 3,
                where_document={"$contains": keyword_query},
            )
        except Exception:
            # Fallback if where_document fails
            keyword_results = None

    # Fusion
    if keyword_results and keyword_results["ids"] and keyword_results["ids"][0]:
        fused = reciprocal_rank_fusion([vector_results, keyword_results])
    else:
        # If keyword search failed, use vector only
        fused = [
            {
                "id": doc_id,
                "score": 1.0 / (1 + dist),  # Convert distance to score
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


def generate_multi_queries(client, model, original_query, question_lang):
    """
    Generate multiple query variations for better retrieval coverage.
    Highest ROI improvement according to RAG best practices.
    """
    if question_lang == "ar":
        instruction = """Generate 3 alternative ways to ask this question in Arabic.
Focus on:
1. More specific technical terms
2. Broader context version
3. Different phrasing with synonyms

Output ONLY the questions, one per line, without numbering."""
    else:
        instruction = """Generate 3 alternative ways to ask this question in English.
Focus on:
1. More specific technical terms
2. Broader context version
3. Different phrasing with synonyms

Output ONLY the questions, one per line, without numbering."""

    try:
        response = chat_completion(
            client,
            model,
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nOriginal Question: {original_query}",
                }
            ],
            temperature=0.8,
        )

        # Parse response
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

        # Clean up (remove numbering if present)
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)
            if line and len(line) > 10:
                cleaned.append(line)

        return cleaned[:MULTI_QUERY_COUNT]
    except Exception as e:
        st.warning(f"Multi-query generation failed: {e}")
        return []


def retrieve_context_enhanced(question, client, model, embedder, collection):
    """
    Enhanced retrieval with:
    1. Multi-query generation
    2. Hybrid search (vector + keyword)
    3. Cross-lingual retrieval
    4. Parent-child chunk handling
    """
    question_lang = detect_language(question)

    # Generate query variations
    query_variations = [question]
    multi_queries = generate_multi_queries(client, model, question, question_lang)
    query_variations.extend(multi_queries)

    # Add cross-lingual query
    translated = translate_query(client, model, question, question_lang)
    if translated:
        query_variations.append(translated)

    # Retrieve with all query variations
    all_results = []
    seen_parent_ids = set()

    for query in query_variations:
        results = hybrid_retrieve(query, collection, embedder, top_k=3)

        for result in results:
            parent_id = result["metadata"].get("parent_id", result["id"])

            # Deduplicate by parent_id (avoid retrieving multiple children from same parent)
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                all_results.append(
                    {
                        "id": result["id"],
                        "score": result["score"],
                        "child_text": result["document"],
                        "parent_text": result["metadata"].get(
                            "parent_text", result["document"]
                        ),
                        "metadata": result["metadata"],
                    }
                )

    # Sort by score and return top-K
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:TOP_K]


# ================================
# ANSWER GENERATION WITH ENHANCED ARABIC SUPPORT
# ================================
def validate_and_clean_arabic(text):
    """
    Validate and clean Arabic responses to ensure proper language.
    """
    # Remove problematic characters
    cleaned = text

    # Remove Chinese characters
    cleaned = re.sub(r"[\u4e00-\u9fff]", "", cleaned)

    # Remove Vietnamese diacritics
    cleaned = re.sub(
        r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Detect and warn about mixed scripts (Latin characters outside parentheses)
    # This regex finds Latin letters NOT inside parentheses
    latin_outside_parens = re.findall(r"(?<!\()[a-zA-Z]{3,}(?![^(]*\))", cleaned)

    if latin_outside_parens and len(latin_outside_parens) > 3:
        # Log warning for excessive mixing
        unique_terms = list(set(latin_outside_parens))[:5]
        st.warning(
            f"⚠️ Language mixing detected. Terms found: {', '.join(unique_terms)}"
        )

    return cleaned


def generate_answer_with_citations(client, model, retrieved_docs, question):
    if not retrieved_docs:
        no_info_ar = "لا تتوفر معلومات ذات صلة في المستندات المرفوعة."
        no_info_en = "I couldn't find relevant information in the uploaded documents."
        return (no_info_ar if detect_language(question) == "ar" else no_info_en), []

    question_lang = detect_language(question)

    # Use PARENT text for generation (full context)
    context_parts = []
    sources = []

    for i, doc in enumerate(retrieved_docs, 1):
        # Use parent text for richer context
        parent_text = doc.get("parent_text", doc["child_text"])
        context_parts.append(f"[Source {i}]:\n{parent_text}")
        sources.append(doc["metadata"])

    context = "\n\n".join(context_parts)

    if question_lang == "ar":
        lang_instruction = """⚠️ LANGUAGE REQUIREMENTS (CRITICAL - MUST FOLLOW):

OUTPUT LANGUAGE: Write ENTIRELY in Arabic (العربية) - Modern Standard Arabic for academic use

TECHNICAL TERMS FORMAT (MANDATORY):
✓ CORRECT: "الرؤية الحاسوبية (Computer Vision)"
✓ CORRECT: "التعرف الضوئي على الحروف (OCR - Optical Character Recognition)"
✗ WRONG: "Computer Vision هو" (mixing scripts in same phrase)
✗ WRONG: "الـ processing" (mixing scripts)
✗ WRONG: "image processing" (all English when should be Arabic)

PUNCTUATION: Use Arabic punctuation marks:
- ، for comma
- ؛ for semicolon  
- . for period
- ؟ for question mark

ABSOLUTELY FORBIDDEN:
❌ No mixing Arabic and Latin scripts in same phrase
❌ No transliteration (writing Arabic words in English letters)
❌ No Chinese, Vietnamese, or other foreign script characters
❌ No answering in English when question is in Arabic
❌ No content not found in the provided sources"""

        examples = """
EXAMPLES OF CORRECT ARABIC RESPONSES:

Good Example 1:
"الرؤية الحاسوبية (Computer Vision) هي مجال متعدد التخصصات يتعامل مع كيفية فهم الحواسيب للصور الرقمية [1]. تُستخدم تقنيات مثل التعرف الضوئي على الحروف (OCR) لتحويل المستندات الممسوحة ضوئياً إلى نص قابل للتحرير [2]."

Good Example 2:
"من التطبيقات الشائعة للرؤية الحاسوبية التعرف على الوجوه (Face Recognition) في الكاميرات الرقمية [1]، واكتشاف الحركة (Motion Detection) في أنظمة المراقبة [2]."

BAD EXAMPLES (DO NOT DO THIS):
❌ "Computer Vision هو مجال مهم" (mixing scripts)
❌ "الرؤية الحاسوبية important جداً" (mixing languages)
❌ "نستخدم image processing" (English technical term not in parentheses)
❌ "الرؤية الحاسوبية تساعدنا" (no citation)
"""
    else:
        lang_instruction = """LANGUAGE: Respond entirely in English.
- Translate any Arabic source content to English
- Use clear, academic language suitable for university students"""
        examples = ""

    system_prompt = f"""You are a university tutor for Computer Vision courses.

{lang_instruction}

CITATIONS (MANDATORY):
- Add [1], [2], [3] etc. immediately after EVERY factual statement
- Multiple sources can be cited together: [1][3]
- If information is not in sources, say: {"'لا تتوفر هذه المعلومة في المواد المتاحة'" if question_lang == "ar" else "'This information is not available in the provided materials'"}

RESPONSE FORMAT:
- Write in clear, flowing paragraphs (no bullet points unless specifically requested)
- Use academic but accessible language
- Be comprehensive yet concise
- Every factual claim must have a citation

QUALITY REQUIREMENTS:
1. {'Every sentence in Arabic only (except English technical terms in parentheses)' if question_lang == 'ar' else 'Every sentence in English only'}
2. Every factual statement must have citation [X]
3. Only use information from the provided sources
4. Maintain academic tone throughout

{examples}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""LECTURE MATERIAL SOURCES:
{context}

STUDENT QUESTION: {question}

Please provide a complete, well-cited answer using ONLY the information from the sources above.""",
        },
    ]

    # Lower temperature for Arabic to reduce hallucination and language mixing
    temperature = 0.4 if question_lang == "ar" else 0.6

    answer = chat_completion(client, model, messages, temperature=temperature)

    # Validate and clean Arabic responses
    if question_lang == "ar":
        answer = validate_and_clean_arabic(answer)

        # Quick quality check
        citations = re.findall(r"\[\d+\]", answer)
        if len(citations) < 2:
            st.warning("⚠️ Response has few citations - answer quality may be limited")

        # Check for excessive Latin script (outside parentheses)
        arabic_chars = sum(1 for c in answer if "\u0600" <= c <= "\u06ff")
        # Count Latin chars outside parentheses (approximate)
        text_outside_parens = re.sub(r"\([^)]*\)", "", answer)
        latin_chars = sum(1 for c in text_outside_parens if "a" <= c.lower() <= "z")

        if arabic_chars > 0 and latin_chars / (arabic_chars + latin_chars) > 0.15:
            st.warning(
                "⚠️ Response may contain mixed languages. Please review carefully."
            )

    return answer, sources


def generate_followups(client, model, question, answer, question_lang):
    if question_lang == "ar":
        instruction = """Based on this Q&A about Computer Vision, generate 3 short follow-up questions in Arabic that a student might ask to deepen understanding.

Focus on:
1. Clarifying concepts mentioned in the answer
2. Asking about related topics
3. Requesting practical examples or applications

Requirements:
- Write questions in pure Arabic (no English mixing)
- Keep questions concise (one sentence each)
- Make questions relevant to Computer Vision course material

Output ONLY the questions, one per line, without numbering."""
    else:
        instruction = """Based on this Q&A about Computer Vision, generate 3 short follow-up questions in English that a student might ask to deepen understanding.

Focus on:
1. Clarifying concepts mentioned in the answer
2. Asking about related topics  
3. Requesting practical examples or applications

Output ONLY the questions, one per line, without numbering."""

    try:
        result = chat_completion(
            client,
            model,
            [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nOriginal Question: {question}\n\nAnswer: {answer[:500]}...",
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


# ================================
# UI RENDERING
# ================================
def render_source_cards(metadatas):
    if not metadatas:
        return

    st.markdown('<div class="section-label">📚 Sources</div>', unsafe_allow_html=True)

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
            <div style="
                background: white;
                border: 1px solid #e5e5e5;
                border-radius: 0.75rem;
                padding: 0.6rem 0.8rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                <div style="
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
                ">{i + 1}</div>
                <div style="overflow: hidden;">
                    <div style="
                        font-weight: 500;
                        font-size: 0.85rem;
                        color: #1a1a1a;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    ">{display_name}</div>
                    <div style="color: #666; font-size: 0.75rem;">{page_label} {page}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_answer(answer, is_rtl=False):
    def replace_citation(match):
        return f'<span class="citation">{match.group(1)}</span>'

    styled = re.sub(r"\[(\d+)\]", replace_citation, answer)
    styled = styled.replace("\n\n", "</p><p>").replace("\n", "<br>")
    styled = f"<p>{styled}</p>"

    css_class = "assistant-message-rtl" if is_rtl else "assistant-message"
    st.markdown(f'<div class="{css_class}">{styled}</div>', unsafe_allow_html=True)


def render_followups(questions, question_lang):
    if not questions:
        return None

    label = "أسئلة ذات صلة" if question_lang == "ar" else "Related Questions"
    st.markdown(f'<div class="section-label">💡 {label}</div>', unsafe_allow_html=True)

    cols = st.columns(len(questions))
    for i, (col, q) in enumerate(zip(cols, questions)):
        with col:
            if st.button(q, key=f"followup_{i}_{hash(q)}", use_container_width=True):
                return q
    return None


def render_message(role, content, sources=None, followups=None, question_lang="en"):
    if role == "user":
        is_rtl = detect_language(content) == "ar"
        direction = 'dir="rtl"' if is_rtl else ""
        st.markdown(
            f'<div class="user-message" {direction}>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        if sources:
            render_source_cards(sources)
        render_answer(content, detect_language(content) == "ar")
        if followups:
            return render_followups(followups, question_lang)
    return None


# ================================
# SIDEBAR
# ================================
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

            total = 0
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    added, status = ingest_document(file, embedder, collection)
                    if status == "exists":
                        st.info(f"✓ {file.name} already indexed")
                    elif status == "success":
                        st.success(f"✓ {file.name}: {added} chunks added")
                        total += added
                    else:
                        st.error(f"✗ {file.name}: {status}")

            if total > 0:
                st.success(f"✅ Total: {total} new chunks indexed")

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
        st.caption(f"• Parent chunks: {PARENT_CHUNK_SIZE} tokens")
        st.caption(f"• Child chunks: {CHILD_CHUNK_SIZE} tokens")
        st.caption(f"• Top-K results: {TOP_K}")
        st.caption(f"• Multi-query count: {MULTI_QUERY_COUNT}")
        st.caption("• Hybrid search: Vector + Keyword")
        st.caption("• Parent-child chunking: Enabled")
        st.caption("• Enhanced Arabic validation: Enabled")

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
                st.success("Database reset!")
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ================================
# MAIN INTERFACE
# ================================
if not st.session_state.chat_history:
    st.markdown(
        """
    <div class="app-header">
        <div class="app-title">🎓 University Tutor - Computer Vision</div>
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
        <span class="provider-badge">🚀 Enhanced RAG: Hierarchical Chunking • Hybrid Search • Multi-Query • Arabic Validation</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

if not st.session_state.api_key:
    st.info("👈 Select a provider and enter your API key in the sidebar to get started")
    st.stop()

# Initialize LLM client
llm_client, llm_model = get_llm_client(
    st.session_state.selected_provider, st.session_state.api_key
)

# Render chat history
followup_clicked = None
for msg in st.session_state.chat_history:
    result = render_message(
        msg["role"],
        msg["content"],
        msg.get("sources"),
        msg.get("followups"),
        msg.get("question_lang", "en"),
    )
    if result:
        followup_clicked = result

if followup_clicked:
    st.session_state.pending_question = followup_clicked
    st.rerun()

# Chat input
question = st.chat_input("Ask anything about Computer Vision...")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    question = question.strip()
    if question:
        question_lang = detect_language(question)

        st.session_state.chat_history.append({"role": "user", "content": question})

        render_message("user", question)

        with st.spinner("🔍 Searching lecture materials and analyzing..."):
            embedder = load_embedder()
            _, collection = get_chroma_collection()

            # Enhanced retrieval with all improvements
            retrieved_docs = retrieve_context_enhanced(
                question, llm_client, llm_model, embedder, collection
            )

            # Generate answer with parent context and Arabic validation
            answer, sources = generate_answer_with_citations(
                llm_client, llm_model, retrieved_docs, question
            )

            # Generate follow-up questions
            followups = generate_followups(
                llm_client, llm_model, question, answer, question_lang
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
    <strong>Enhanced RAG Architecture:</strong> Hierarchical Parent-Child Chunking • Hybrid Search (Vector + Keyword) •
    Multi-Query Generation • Cross-lingual Retrieval • Arabic Language Validation<br>
    <strong>Powered by:</strong> {provider_name} • ChromaDB • E5-multilingual-small
</div>
""",
    unsafe_allow_html=True,
)
