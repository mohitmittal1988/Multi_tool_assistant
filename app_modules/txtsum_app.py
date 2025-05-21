def run(api_key):
    import validators
    import streamlit as st
    from langchain.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain.chains.summarize import load_summarize_chain
    from langchain_community.document_loaders import UnstructuredURLLoader
    import re
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain.docstore.document import Document


    st.title("🦜 TEXT SUMMARIZER 🦜")
    st.subheader("Summarize from YouTube or Website")



    generic_url = st.text_input("Enter a YouTube or Website URL", label_visibility="visible")

    if st.button("Summarize"):
        if not api_key.strip() or not generic_url.strip():
            st.error("Please provide the API key and a valid URL.")
            return

        if not validators.url(generic_url):
            st.error("Invalid URL format.")
            return

        llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)

        prompt_template = PromptTemplate(
            template="""
            Provide a summary of the following content in 300 words:
            Content:{text}
            """,
            input_variables=["text"]
        )

        docs = []
        try:
            if "youtube.com" in generic_url:
                match = re.search(r"v=([a-zA-Z0-9_-]+)", generic_url)
                if not match:
                    st.error("Unable to extract video ID from YouTube URL.")
                    return
                video_id = match.group(1)
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([entry["text"] for entry in transcript])
                docs = [Document(page_content=full_text[:12000])]
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                docs = loader.load()
                if docs:
                    docs[0].page_content = docs[0].page_content[:12000]

            if docs:
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run(docs)
                st.success(summary)
            else:
                st.error("No content found to summarize.")

        except Exception as e:
            st.exception(f"Error: {e}")
