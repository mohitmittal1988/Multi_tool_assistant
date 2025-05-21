# 🤖 AI Multi-Tool Assistant

A modular and intelligent Streamlit app that offers four powerful AI tools in one place:

- 🦜 **Text Summarizer** (YouTube & Web)
- 🧮 **Math & Reasoning Solver**
- 🔎 **Agentic Web Search** (Wikipedia, Arxiv, DuckDuckGo)
- 📄 **Conversational PDF RAG**

All powered by **Groq’s blazing-fast LLaMA3 & Gemma models** using LangChain.

---

## 🚀 Features

- 🎯 Summarize YouTube videos and websites with one click
- 🧮 Solve math and logic problems using LLM reasoning
- 🌐 Perform real-time research via Wikipedia, Arxiv, and web search
- 📄 Ask questions about PDF content using Retrieval-Augmented Generation (RAG)
- ⚡ Built on top of `Groq + LangChain` for ultra-fast LLM inference
- 🧠 Maintains multi-turn chat memory per tool

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
- [ChromaDB](https://docs.trychroma.com/)
- [HuggingFace Embeddings](https://huggingface.co/)
- Python 3.10+

---

## 📁 Folder Structure





### 1. Clone the Repository

  ```bash
  git clone https://github.com/mohitmittal1988/Multi_tool_assistant.git
  cd text_summary
  ```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the App

```bash
streamlit run main.py
```

---

## 🔐 API Key Requirement

When prompted in the UI, provide your **Groq API Key** to access the `Llama3-8b-8192` LLM.

---

## 📁 Project Structure

```plaintext
Combined_Assistant/
├── main.py # Central app with sidebar navigation
├── app_modules/ # Modular sub-apps
│ ├── txtsum_app.py # YouTube/Web summarizer
│ ├── math_solver_app.py # Math & logic solver
│ ├── agentic_search_app.py # LLM agent using Wikipedia, Arxiv, Search
│ └── pdf_rag_chat_app.py # Chat with PDF using Conversational RAG
├── requirements.txt
└── README.md
```

---

