import streamlit as st
from css import CUSTOM_CSS
from state import init_session_state
from ui import Ui
from config import Config
from llm_client import LLMClient
from languageUtils import LanguageUtils
from loaders import load_embedder
from chroma_utils import get_chroma_collection
from rag_pipeline import RAGPipeline

def main():
    st.set_page_config(
    page_title="RAG Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    init_session_state()
    """Main application."""
    Ui.render_sidebar()

    # Welcome
    if not st.session_state.chat_history:
        st.markdown(
            """
        <div class="app-header">
            <div class="app-title">🎓 RAG Tutor</div>
            <div class="app-subtitle">Ask questions about your lecture materials in English or Arabic</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        provider = Config.LLM_PROVIDERS[st.session_state.selected_provider]["name"]
        st.markdown(
            f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="provider-badge">⚡ Powered by {provider}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if not st.session_state.api_key:
        st.info("👈 Enter your API key in the sidebar to get started")
        st.stop()

    # Initialize client
    llm_client = LLMClient(st.session_state.selected_provider, st.session_state.api_key)

    # Render history
    followup_clicked = None
    for i, msg in enumerate(st.session_state.chat_history):
        result = Ui.render_message(
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
            Ui.render_message("user", question)

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

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "followups": followups,
                    "lang": lang,
                }
            )
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <strong>Features:</strong> Intelligent Query Analysis • Multi-Strategy Retrieval • 
        Cross-Encoder Reranking • Strict Grounding • Bilingual Support
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
