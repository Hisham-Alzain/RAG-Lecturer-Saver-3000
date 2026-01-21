import streamlit as st
from collections import OrderedDict
from config import Config

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "chat_history": [],
        "api_key": "",
        "selected_provider": list(Config.LLM_PROVIDERS.keys())[0],
        "semantic_cache": OrderedDict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value