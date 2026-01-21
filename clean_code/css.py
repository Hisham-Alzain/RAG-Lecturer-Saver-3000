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