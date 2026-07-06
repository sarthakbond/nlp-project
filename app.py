import tempfile
import os
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

from src.pdf_utils import extract_text
from src.text_utils import clean_text, chunk_text
from src.embeddings import create_embeddings
from src.vector_store import build_index
from src.rag import retrieve
from src.llm import load_model, generate_answer


# Page Config

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chat Assistant")

st.markdown("""
### Stop scrolling. Start asking!

Turn static documents into interactive conversations.

Upload your PDF, ask questions, and let AI instantly find the most relevant information for you!

*Powered by Retrieval-Augmented Generation (RAG), Sentence Transformers, FAISS, and Google Gemini.*
""")

st.divider()

# API Key

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Session State

if "processed" not in st.session_state:
    st.session_state.processed = False

# Upload PDF

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None and not st.session_state.processed:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    with st.spinner("Processing PDF..."):

        # Extract text
        text = extract_text(pdf_path)

        # Clean text
        text = clean_text(text)

        # Chunk text
        chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        # Embedding model
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Embeddings
        embeddings = create_embeddings(
            chunks,
            embedding_model
        )

        # FAISS
        index = build_index(
            embeddings
        )

        # Gemini
        llm = load_model(API_KEY)

    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.embedding_model = embedding_model
    st.session_state.llm = llm
    st.session_state.processed = True

    st.success("✅ PDF processed successfully!")

# Ask Questions

if st.session_state.processed:

    query = st.text_input(
        "Ask a question about the PDF"
    )

    if st.button("Ask"):

        if query.strip():

            with st.spinner("Thinking..."):

                context, _ = retrieve(
                    query=query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    model=st.session_state.embedding_model
                )

                answer = generate_answer(
                    st.session_state.llm,
                    query,
                    context
                )

            st.subheader("Answer")
            st.write(answer)

        else:
            st.warning("Please enter a question.")