def run(api_key):
    import streamlit as st
    from langchain_groq import ChatGroq
    from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
    from langchain.agents import initialize_agent, AgentType
    from langchain.callbacks import StreamlitCallbackHandler

    # UI Header
    st.title("🔎 Agentic Search Assistant")
    st.markdown("Ask anything. This agent searches the web, Wikipedia, and Arxiv using Groq's LLM.")

    # ✅ Use passed API key instead of requesting it again
    if not api_key:
        st.warning("Please provide your Groq API key.")
        return

    # Load tools
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200), name="Arxiv")
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200), name="Wikipedia")
    search = DuckDuckGoSearchRun(name="Search")
    tools = [search, arxiv, wiki]

    # Load LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        agent_kwargs={"max_iterations": 3}
    )

    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can look things up from Arxiv, Wikipedia, or the web. Ask me anything!"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(user_input, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
