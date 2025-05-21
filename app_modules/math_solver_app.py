def run(api_key):
    import streamlit as st
    from langchain_groq import ChatGroq
    from langchain.chains import LLMMathChain, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.callbacks import StreamlitCallbackHandler

    # ✅ DO NOT call st.set_page_config() here (already set in main.py)
    st.title("🧮 Math Problem & Logic Solver")
    st.subheader("Ask a math or reasoning question")

    # ✅ Use the passed `api_key`
    if not api_key:
        st.warning("Please provide your Groq API Key.")
        return

    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # Define tools
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=WikipediaAPIWrapper().run,
        description="For finding general knowledge information"
    )

    math_chain = LLMMathChain.from_llm(llm=llm)
    calculator_tool = Tool(
        name="Calculator",
        func=math_chain.run,
        description="For solving math questions using expressions"
    )

    reasoning_prompt = PromptTemplate(
        template="""
        You are a math reasoning assistant. Solve the question logically and explain step by step:
        Question: {question}
        Answer:
        """,
        input_variables=["question"]
    )
    reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)
    reasoning_tool = Tool(
        name="Reasoning Tool",
        func=reasoning_chain.run,
        description="For logic or word problem solving"
    )

    # Combine tools into an agent
    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator_tool, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Set up session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your math assistant. Ask me a question!"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Ask a question
    question = st.text_area("Enter your math or logic question:")

    if st.button("Get Answer"):
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.spinner("Solving..."):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.success(response)
        else:
            st.warning("Please enter a question.")
