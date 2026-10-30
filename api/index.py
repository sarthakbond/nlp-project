import time
import re
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="LexiQA Legal & Compliance Intelligence QA Engine",
    description="Topic #9: Question Answering System via TF-IDF and Cosine Similarity",
    version="1.0.0"
)

# Enable CORS for local testing and Vercel cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    document_text: Optional[str] = ""
    chunks: Optional[List[str]] = []
    top_k: Optional[int] = 3


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Break legal text into overlapping chunks or by clauses/paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    if len(paragraphs) >= 3:
        # Paragraph/Clause-based chunking is ideal for legal documents
        return paragraphs

    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def extract_extractive_answer(question: str, context: str, top_sentences: int = 2) -> str:
    """Topic #9: Extractive QA using TF-IDF sentence ranking."""
    if not context or not context.strip():
        return "No relevant legal clause found in this contract."

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return context[:350] + ("..." if len(context) > 350 else "")

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf_sentences = vec.fit_transform(sentences)
        query_vec = vec.transform([question])
        scores = cosine_similarity(query_vec, tfidf_sentences).flatten()

        top_indices = np.argsort(scores)[::-1][:top_sentences]
        best_sentences = [sentences[i] for i in top_indices if scores[i] > 0]

        if not best_sentences:
            return sentences[0]
        return " ".join(best_sentences)
    except Exception:
        return sentences[0]


def assess_clause_risk(clause_text: str) -> dict:
    """Identifies potential legal risk levels based on clause terms."""
    lower = clause_text.lower()
    high_risk_triggers = ["uncapped liability", "indemnify and hold harmless", "liquidated damages", "immediate termination", "penalties", "gross negligence", "irrevocable", "perpetual and exclusive"]
    moderate_risk_triggers = ["limitation of liability", "governing law", "jurisdiction", "audit rights", "non-compete", "confidentiality", "warranty disclaimer", "intellectual property assignment"]

    found_high = [t for t in high_risk_triggers if t in lower]
    found_mod = [t for t in moderate_risk_triggers if t in lower]

    if found_high:
        return {"level": "HIGH RISK", "color": "#dc2626", "badge": "Attention Critical", "triggers": found_high}
    elif found_mod:
        return {"level": "MODERATE", "color": "#d97706", "badge": "Review Advised", "triggers": found_mod}
    return {"level": "LOW RISK", "color": "#16a34a", "badge": "Standard Provision", "triggers": []}


@app.get("/")
@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "engine": "Python Scikit-Learn TF-IDF Core",
        "topic": "Topic #9: Question Answering System",
        "curriculum": "Keystone School of Engineering - Dept of AI & Data Science",
        "algorithm": "TF-IDF + Cosine Similarity",
        "team": {
            "group": "Skill G",
            "members": [
                "Sarthak Phadatare",
                "Pranit Latpate",
                "Aqib Pansare",
                "Shamita Marathe"
            ]
        }
    }


@app.post("/api/query")
def process_query(req: QueryRequest):
    start_time = time.time()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chunks = req.chunks or []
    if not chunks and req.document_text:
        chunks = chunk_text(req.document_text)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document text or chunks required.")

    # 1. Topic #9 TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([query])

    # 2. Topic #9 Cosine Similarity Computation
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 3. Ranking
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_k = min(req.top_k or 3, len(chunks))
    top_indices = sorted_indices[:top_k]

    retrieved = []
    all_scores = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(similarity_scores[idx])
        chunk_content = chunks[idx]
        all_scores.append(score)
        risk = assess_clause_risk(chunk_content)
        retrieved.append({
            "rank": rank,
            "chunk_index": int(idx),
            "score": round(score, 4),
            "content": chunk_content,
            "risk": risk
        })

    # Top retrieved text as context
    best_chunk = chunks[top_indices[0]] if len(top_indices) > 0 else ""
    context_text = "\n\n".join([r["content"] for r in retrieved])
    extracted_answer = extract_extractive_answer(query, context_text)

    # Compute top terms
    feature_names = vectorizer.get_feature_names_out()
    mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_term_indices = np.argsort(mean_weights)[::-1][:12]
    top_terms = [feature_names[i] for i in top_term_indices]

    # Query tokens found in vocabulary
    query_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()) if w in vectorizer.vocabulary_]

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "engine": "Python Scikit-Learn Serverless Engine",
        "query": query,
        "answer": extracted_answer,
        "retrieved_passages": retrieved,
        "stats": {
            "total_chunks": len(chunks),
            "vocab_size": len(feature_names),
            "latency_ms": latency_ms,
            "top_terms": top_terms,
            "matched_query_tokens": query_tokens,
            "primary_similarity_score": round(float(similarity_scores[top_indices[0]]), 4) if len(top_indices) > 0 else 0.0
        }
    }
