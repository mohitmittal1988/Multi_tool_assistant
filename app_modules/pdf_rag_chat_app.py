def run(api_key):
    import streamlit as st
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_chroma import Chroma
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_groq import ChatGroq
    from langchain_core.runnables import RunnableWithMessageHistory
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    import os

    # Initialize embedding model
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # UI
    st.title("📄 Chat with Your PDF")
    st.write("Upload a PDF and ask questions about its content.")

    session_id = st.text_input("Session ID", value="default-session")

    if not api_key:
        st.warning("Please provide your Groq API key in the sidebar.")
        return

    # Load PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.error("No text could be extracted from the PDF.")
            return

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        if not splits:
            st.error("Failed to split text.")
            return

        # Build vectorstore
        try:
            vectorstore = Chroma.from_documents(splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
        except Exception as e:
            st.error(f"Error during vectorstore creation: {e}")
            return

        # History-aware retriever
        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the user's question based on chat history. Only rephrase, do not answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(
            ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it"),
            retriever,
            contextual_prompt
        )

        # Answering chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using the context below. Be concise. \n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it"), qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Chat state manager
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if "store" not in st.session_state:
                st.session_state.store = {}
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Chat input
        user_input = st.text_input("Ask something about the PDF:")
        if user_input:
            session_history = get_session_history(session_id)
            chain = RunnableWithMessageHistory(
                rag_chain,
                lambda _: session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
            st.success(response["answer"])
            st.write("🧠 Chat History:", session_history.messages)
