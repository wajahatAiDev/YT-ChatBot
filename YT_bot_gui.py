import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()  # still works locally with a .env file

try:
    os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
except Exception:
    pass 


st.set_page_config(page_title="Ask a YouTube Video", page_icon="🎬")
st.title("🎬 Ask a Video")


def text_clean(docs):
    return "\n\n".join(chunk.page_content for chunk in docs)


@st.cache_resource(show_spinner=False)
def build_retriever(video_id: str):
    """Fetch transcript, split, embed and index it. Cached per video ID."""
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def build_chain(retriever):
    prompt = PromptTemplate(
        template="""You are a helpful assistant.Answer ONLY from the provided transcript context. If the context is insufficient, just say you don not know. {context} Question: {question} """,
    input_variables=["context", "question"]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(text_clean),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | prompt | llm | StrOutputParser()


video_id = st.text_input("Video ID", placeholder="e.g. dQw4w9WgXcQ")
question = st.text_input("Your question about the video", placeholder="Can you summarize the video?")

if st.button("Ask", type="primary"):
    if not video_id.strip() or not question.strip():
        st.warning("Please enter both a video ID and a question.")
    else:
        try:
            with st.spinner("Processing transcript..."):
                retriever = build_retriever(video_id.strip())
        except Exception as e:
            st.error(f"Could not get or process the transcript: {e}")
            st.stop()

        try:
            with st.spinner("Thinking..."):
                answer = build_chain(retriever).invoke(question.strip())
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating the answer: {e}")
