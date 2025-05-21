import streamlit as st

# Import all app modules
from app_modules import txtsum_app, math_solver_app, agentic_search_app, pdf_rag_chat_app
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


st.set_page_config(page_title="🧠 All-in-One AI Assistant", layout="wide")
st.title("🤖 AI Multi-Tool Assistant")
st.markdown("""
Welcome! This app combines multiple AI-powered tools:
- 🦜 Text Summarizer (YouTube, Website)
- 🧮 Math & Reasoning Solver
- 🔎 Web Search with Wikipedia/Arxiv/DuckDuckGo
- 📄 Conversational PDF RAG
""")

# Sidebar navigation
st.sidebar.title("🔧 Choose a Tool")
tool = st.sidebar.radio("Select Functionality", [
    "Text Summarizer",
    "Math & Reasoning Solver",
    "Web Search Assistant",
    "Chat with PDF (RAG)"
])

# Shared API key
api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")

# Route to selected app
if tool == "Text Summarizer":
    txtsum_app.run(api_key)
elif tool == "Math & Reasoning Solver":
    math_solver_app.run(api_key)
elif tool == "Web Search Assistant":
    agentic_search_app.run(api_key)
elif tool == "Chat with PDF (RAG)":
    pdf_rag_chat_app.run(api_key)
