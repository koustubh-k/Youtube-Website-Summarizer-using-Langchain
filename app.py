import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from warnings import filterwarnings

# Suppress all warnings for a cleaner Streamlit UI
filterwarnings("ignore")

# Streamlit App Setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦉 LangChain: Summarize Text From YT or Website")
st.markdown("Enter a YouTube video URL or a website URL below to get a concise summary using a large language model.")

# Sidebar for API key input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

# URL Input
generic_url = st.text_input("Enter YouTube or Web Article URL", label_visibility="visible")

# Prompt Template for Summarization
refine_template = """
You are a master summarizer. Your task is to provide a comprehensive and detailed summary of the following content.
Your summary should be at least 300 words long and capture all the key points, arguments, and conclusions presented.

Content:
{text}
"""

prompt = PromptTemplate(template=refine_template, input_variables=["text"])

# Check if Summarize button is clicked
if st.button("Summarize the Content"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the Groq API key and a URL to get started.")
    
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    
    else:
        try:
            with st.spinner("⏳ Loading content and summarizing..."):

                # Load the document
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Brave/117.0.0.0"
                        }
                    )

                docs = loader.load()

                # Initialize LLM
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

                # Load summarize chain
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt,
                    verbose=True
                )

                # Get the summary
                result = chain.invoke({"input_documents": docs})

                st.success("✅ Summary created successfully!")
                st.markdown(result['output_text'])

        except Exception as e:
            st.exception(f"An error occurred: {e}")
