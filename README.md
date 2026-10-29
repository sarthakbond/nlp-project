# 📄 PDF Question Answering System (Topic #9)

A comprehensive Natural Language Processing (NLP) application designed to ingest PDF documents, process natural language queries, and retrieve relevant answers based on **Topic #9: Question Answering System** curriculum requirements.

---

## 🎯 Topic #9 Curriculum Alignment

| Dimension | Rubric Requirement | Implementation in this Project |
| :--- | :--- | :--- |
| **Core Objective** | Question Answering System based on given document | Ingests PDF documents and answers natural language queries |
| **Methodology** | Question Processing, Information Retrieval | Preprocessing, Query Vectorization, Chunk Similarity Ranking |
| **Mandated Algorithm** | **TF-IDF + Cosine Similarity** | **`TfidfVectorizer` (scikit-learn) + `cosine_similarity`** |
| **Advanced Stack (Optional)** | Modern Transformer RAG comparison | Dense Embeddings (`all-MiniLM-L6-v2`) + **FAISS** + **Gemini 2.5 Flash** |

---

## 🔬 Mathematical Formulation

### 1. Term Frequency – Inverse Document Frequency (TF-IDF)
For each term $t$ in a document chunk $d$ across a corpus of chunks $D$:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2. Cosine Similarity for Information Retrieval
Given the vectorized user query $\mathbf{q}$ and document chunk vector $\mathbf{d}$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

Document chunks are ranked descending by their cosine similarity scores, and the top-$k$ most relevant passages are selected as context.

---

## 🚀 Key Features

* **Strict Syllabus Compliance (Topic #9):** Uses TF-IDF vectorization and Cosine Similarity for document retrieval.
* **Offline Extractive QA Mode:** Can extract answers directly from the highest-ranked text passages without needing external cloud APIs.
* **Optional Generative AI Synthesis:** Connects with Google Gemini (`gemini-2.5-flash`) when an API key is provided for conversational answers.
* **Cosine Similarity Inspector:** Displays exact numerical similarity scores for all retrieved document chunks directly in the Streamlit UI.
* **Document Vocabulary Insights:** Automatically extracts and displays top keywords and vocabulary size computed from the TF-IDF matrix.
* **Dual-Architecture Support:** Easy toggle between **Topic #9 Classical NLP** and **Modern Dense RAG (FAISS + MiniLM)** for comparison during presentations and viva evaluations.

---

## 📁 Project Structure

```text
NLP project/
├── app.py                  # Interactive Streamlit Web Interface
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation & Rubric Mapping
├── data/
│   └── sample.pdf          # Sample reference PDF for testing
├── notebooks/              # Jupyter Notebooks for experimentation
│   ├── 01_read_pdf.ipynb
│   ├── 02_chunk_text.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_faiss_similarity_search.ipynb
│   └── 05_rag_pipeline.ipynb
└── src/
    ├── config.py           # Configuration (chunk size, overlap, top-k)
    ├── pdf_utils.py        # PDF text extraction (pypdf + PyMuPDF fallback)
    ├── text_utils.py       # Text cleaning & character/token chunking
    ├── embeddings.py       # TF-IDF Vectorizer & dense embedding logic
    ├── vector_store.py     # Cosine Similarity ranking & FAISS index
    ├── rag.py              # Information retrieval pipeline
    └── llm.py              # Extractive QA & Gemini answer synthesis
```

---

## 💻 Installation & Quickstart

### 1. Activate Environment
In PowerShell:
```powershell
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Set up Gemini API Key
Create a `.env` file in the project root:
```text
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: If you do not have an API key, the system automatically uses **Extractive QA**, so it runs 100% offline!)*

### 4. Run the Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
