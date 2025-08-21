import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize LLM
if groq_api_key.strip():
    llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
else:
    llm = None

# Prompt
prompt_template = """
Provide a clear, concise summary of the following content in about 300 words:
Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button action
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both API Key and URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or website link.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                if not docs:
                    st.error("No content could be loaded from the given URL.")
                else:
                    # Summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(input_documents=docs)
                    st.success(output_summary)

        except Exception as e:
            st.error(f"Exception: {e}")

