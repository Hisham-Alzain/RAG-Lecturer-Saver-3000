import re
import streamlit as st
from collections import  OrderedDict
from typing import List, Dict, Optional
from languageUtils import LanguageUtils
from config import Config
from document_ingestor import DocumentIngester
from loaders import load_embedder
from chroma_utils import get_chroma_collection

class Ui:
    """UI rendering components."""

    @staticmethod
    def render_sources(sources: List[Dict]):
        """Render source cards."""
        if not sources:
            return

        st.markdown(
            '<div class="section-label">📚 Sources</div>', unsafe_allow_html=True
        )

        cols = st.columns(min(len(sources), 3))
        for i, (col, src) in enumerate(zip(cols * ((len(sources) // 3) + 1), sources)):
            with cols[i % 3]:
                name = src.get("source", "Unknown")
                page = src.get("page", "?")
                display_name = name[:25] + "..." if len(name) > 25 else name

                st.markdown(
                    f"""
                <div class="source-card">
                    <div class="source-number">{i + 1}</div>
                    <div>
                        <div style="font-weight: 500; font-size: 0.85rem;">{display_name}</div>
                        <div style="color: #666; font-size: 0.75rem;">Page {page}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    @staticmethod
    def render_answer(answer: str, is_rtl: bool = False):
        """Render answer with styled citations."""

        def replace_citation(match):
            return f'<span class="citation">{match.group(1)}</span>'

        styled = re.sub(r"\[(\d+)\]", replace_citation, answer)
        styled = styled.replace("\n\n", "</p><p>").replace("\n", "<br>")
        styled = f"<p>{styled}</p>"

        css_class = "assistant-message-rtl" if is_rtl else "assistant-message"
        st.markdown(f'<div class="{css_class}">{styled}</div>', unsafe_allow_html=True)

    @staticmethod
    def render_followups(
        questions: List[str], lang: str, msg_idx: int
    ) -> Optional[str]:
        """Render follow-up question buttons."""
        if not questions:
            return None

        label = "أسئلة ذات صلة" if lang == "ar" else "Related Questions"
        st.markdown(
            f'<div class="section-label">💡 {label}</div>', unsafe_allow_html=True
        )

        cols = st.columns(len(questions))
        for i, (col, q) in enumerate(zip(cols, questions)):
            with col:
                if st.button(
                    q, key=f"followup_{msg_idx}_{i}", use_container_width=True
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
        lang: str = "en",
        msg_idx: int = 0,
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
                list(Config.LLM_PROVIDERS.keys()),
                index=list(Config.LLM_PROVIDERS.keys()).index(st.session_state.selected_provider),
                label_visibility="collapsed",
            )
            st.session_state.selected_provider = selected
    
            config = Config.LLM_PROVIDERS[selected]
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
    
    
    