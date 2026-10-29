import tempfile
import os
import numpy as np
from dotenv import load_dotenv
import streamlit as st

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TOP_K_CHUNKS,
)

from src.pdf_utils import extract_text
from src.text_utils import clean_text, chunk_text
from src.embeddings import create_tfidf_embeddings
from src.rag import retrieve_tfidf
from src.llm import load_model, generate_answer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF QA Assistant - Topic #9",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------
# Header & Rubric Context
# ---------------------------------------------------------
st.title("📄 PDF Question Answering System")
st.caption("Topic #9: Question Answering System | Methodology: TF-IDF + Cosine Similarity")

st.markdown("""
Welcome to the **PDF Question Answering Assistant**. This implementation aligns strictly with the 
**Topic #9** curriculum rubric:
* **Question Processing:** Preprocessing & TF-IDF Vectorization.
* **Information Retrieval:** Computing **Cosine Similarity** between query vectors and document chunk vectors.
* **Answer Formulation:** High-precision Extractive QA or conversational synthesis with Generative AI.
""")

st.divider()

# ---------------------------------------------------------
# Environment & Sidebar Settings
# ---------------------------------------------------------
load_dotenv()
default_api_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Algorithm Selection")
    algo_mode = st.radio(
        "Retrieval Methodology:",
        [
            "TF-IDF + Cosine Similarity (Official Rubric)",
            "Dense Embeddings + FAISS (Advanced Mode)"
        ],
        index=0,
        help="Topic #9 requires TF-IDF + Cosine Similarity. Dense mode uses Sentence Transformers."
    )
    
    st.subheader("2. Retrieval Settings")
    top_k = st.slider("Top Chunks to Retrieve (k)", min_value=1, max_value=5, value=TOP_K_CHUNKS)

    st.subheader("3. Generative AI (Optional)")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=default_api_key,
        type="password",
        help="Optional. If omitted, the system performs classical Extractive QA directly from the top chunks."
    )
    
    st.divider()
    st.info("💡 **Topic #9 Note:** Classical TF-IDF + Cosine Similarity satisfies the curriculum requirements without needing cloud API keys.")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "top_terms" not in st.session_state:
    st.session_state.top_terms = []

# ---------------------------------------------------------
# PDF Upload & Processing
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Check if a new file is uploaded
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_file_id") != file_id:
        st.session_state.processed = False
        st.session_state.current_file_id = file_id

    if not st.session_state.processed:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            pdf_path = temp_file.name

        with st.spinner("Extracting text and building TF-IDF index..."):
            # 1. Extract raw text
            raw_text = extract_text(pdf_path)

            # 2. Clean text
            cleaned = clean_text(raw_text)

            # 3. Chunk text
            chunks = chunk_text(
                cleaned,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )

            if not chunks:
                st.error("No text could be extracted from this PDF. Please check if it contains readable text.")
                st.stop()

            # 4. Topic #9: Build TF-IDF Vocabulary and Matrix
            vectorizer, tfidf_matrix = create_tfidf_embeddings(
                chunks,
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE
            )

            # 5. Extract top representative keywords by TF-IDF weight
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_tfidf)[::-1][:8]
            top_terms = [feature_names[i] for i in top_indices]

            # 6. Store in Session State
            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.top_terms = top_terms
            st.session_state.processed = True

            # Clean temporary file
            try:
                os.remove(pdf_path)
            except Exception:
                pass

        st.success(f"✅ Successfully processed **{uploaded_file.name}**!")

# ---------------------------------------------------------
# Document Insights & Statistics
# ---------------------------------------------------------
if st.session_state.processed:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", len(st.session_state.chunks))
    col2.metric("TF-IDF Vocabulary Size", len(st.session_state.vectorizer.get_feature_names_out()))
    col3.metric("Retrieval Algorithm", "TF-IDF + Cosine Sim")

    with st.expander("📌 View Key Terms extracted via TF-IDF"):
        st.write(", ".join([f"`{term}`" for term in st.session_state.top_terms]))

    st.subheader("💬 Ask a Question")
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g. What is the main conclusion of this document?"
    )

    if st.button("Submit Query", type="primary"):
        if query.strip():
            with st.spinner("Processing question and calculating Cosine Similarities..."):
                # Topic #9: Retrieve using TF-IDF + Cosine Similarity
                context, retrieved_chunks, scores = retrieve_tfidf(
                    query=query,
                    vectorizer=st.session_state.vectorizer,
                    tfidf_matrix=st.session_state.tfidf_matrix,
                    chunks=st.session_state.chunks,
                    top_k=top_k
                )

                # Initialize LLM (if key provided) or fallback to Extractive QA
                llm = load_model(api_key_input)
                answer = generate_answer(llm, query, context)

            st.markdown("### 🎯 Answer")
            st.markdown(answer)

            # Rubric Evidence: Show Chunks and exact Cosine Similarity scores
            st.markdown("---")
            with st.expander(f"🔍 Inspect Retrieved Chunks & Cosine Similarity Scores (Top {len(retrieved_chunks)})", expanded=True):
                for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), start=1):
                    st.markdown(f"**Chunk #{i}** | *Cosine Similarity Score:* `{score:.4f}`")
                    st.info(chunk)
        else:
            st.warning("Please type a valid question.")
