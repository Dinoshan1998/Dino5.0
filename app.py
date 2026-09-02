import os
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from groq import Groq
from tavily import TavilyClient


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================================================
# CHECK API KEYS
# =========================================================

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env")

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY not found in .env")


# =========================================================
# CLIENTS
# =========================================================

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF + Web AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 PDF + Web Search AI Chatbot")

st.write(
    "Upload a PDF and ask questions. "
    "The chatbot will first search the PDF and can also use web search."
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def create_chunks(text, chunk_size=800, overlap=150):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# CREATE FAISS VECTOR DATABASE
# =========================================================

def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# =========================================================
# SEARCH PDF
# =========================================================

def search_pdf(question, top_k=4):

    if st.session_state.index is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, indices = st.session_state.index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i != -1:

            results.append(
                st.session_state.chunks[i]
            )

    return results


# =========================================================
# WEB SEARCH
# =========================================================

def web_search(question):

    response = tavily_client.search(
        question,
        search_depth="basic",
        max_results=5,
        include_answer=True
    )

    results = response.get("results", [])

    web_context = []

    for result in results:

        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        web_context.append(
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}"
        )

    return "\n\n".join(web_context)


# =========================================================
# ASK GROQ
# =========================================================

def ask_llm(question, pdf_context, web_context):

    system_prompt = """
You are a helpful AI assistant.

You have two possible information sources:

1. PDF CONTEXT
2. WEB SEARCH CONTEXT

Rules:

- First use the PDF context.
- If the PDF contains enough information to answer the question,
  answer using the PDF.
- If the PDF does not contain the answer, use the web search context.
- Clearly say when information comes from the web.
- Do not invent facts.
- If neither source contains enough information, say that you
  could not find enough information.
- Give a clear and concise answer.
"""

    user_prompt = f"""
QUESTION:
{question}

========================
PDF CONTEXT
========================

{pdf_context}

========================
WEB SEARCH CONTEXT
========================

{web_context}

========================

Answer the question using the available context.
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 PDF Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("🔄 Process PDF"):

            with st.spinner(
                "Reading and processing PDF..."
            ):

                pdf_text = extract_pdf_text(
                    uploaded_file
                )

                if not pdf_text.strip():

                    st.error(
                        "❌ Could not extract text from PDF."
                    )

                else:

                    chunks = create_chunks(
                        pdf_text
                    )

                    index = create_vector_database(
                        chunks
                    )

                    st.session_state.chunks = chunks
                    st.session_state.index = index

                    st.success(
                        f"✅ PDF processed! "
                        f"{len(chunks)} chunks created."
                    )

    st.divider()

    st.header("🌐 Search Settings")

    use_web = st.checkbox(
        "Enable Web Search",
        value=True
    )

    st.info(
        "PDF is searched first. "
        "Web search is used for additional information."
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your PDF or the web..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # Display user message

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------------
    # SEARCH PDF
    # -----------------------------------------------------

    pdf_results = search_pdf(
        question,
        top_k=4
    )

    pdf_context = "\n\n".join(
        pdf_results
    )

    # -----------------------------------------------------
    # WEB SEARCH
    # -----------------------------------------------------

    web_context = ""

    if use_web:

        with st.spinner(
            "🌐 Searching the web..."
        ):

            try:

                web_context = web_search(
                    question
                )

            except Exception as e:

                st.warning(
                    f"Web search failed: {e}"
                )

    # -----------------------------------------------------
    # ASK AI
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Thinking..."
        ):

            try:

                answer = ask_llm(
                    question,
                    pdf_context,
                    web_context
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    f"❌ Error while generating answer:\n\n{e}"
                )

                st.error(answer)

    # Save answer

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )