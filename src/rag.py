import numpy as np
from src.vector_store import search_tfidf, search_index


def retrieve_tfidf(query, vectorizer, tfidf_matrix, chunks, top_k=3):
    """
    Topic #9 Information Retrieval:
    Retrieves the most relevant text chunks using TF-IDF + Cosine Similarity.

    Returns
    -------
    context : str
        Concatenated text of retrieved chunks.
    retrieved_chunks : list[str]
        List of selected text chunks.
    scores : list[float]
        Cosine similarity scores for each retrieved chunk.
    """
    scores, indices = search_tfidf(vectorizer, tfidf_matrix, query, top_k=top_k)

    retrieved_chunks = []
    retrieved_scores = []

    for score, idx in zip(scores, indices):
        # Keep non-zero matches, or at least the top chunk if everything is zero
        if score > 0.0 or len(retrieved_chunks) == 0:
            retrieved_chunks.append(chunks[idx])
            retrieved_scores.append(float(score))

    context = "\n\n".join(retrieved_chunks)
    return context, retrieved_chunks, retrieved_scores


def retrieve_dense(query, index, chunks, model, top_k=3):
    """
    Dense retrieval using Sentence Transformers and FAISS index.
    """
    query_embedding = model.encode([query])
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    query_embedding = np.ascontiguousarray(query_embedding)

    distances, indices = search_index(index, query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    context = "\n\n".join(retrieved_chunks)
    return context, retrieved_chunks, distances[0].tolist()


def retrieve(query, **kwargs):
    """
    Unified retrieval interface supporting both TF-IDF and Dense modes.
    """
    if "vectorizer" in kwargs and "tfidf_matrix" in kwargs:
        return retrieve_tfidf(
            query=query,
            vectorizer=kwargs["vectorizer"],
            tfidf_matrix=kwargs["tfidf_matrix"],
            chunks=kwargs["chunks"],
            top_k=kwargs.get("top_k", 3)
        )
    else:
        return retrieve_dense(
            query=query,
            index=kwargs["index"],
            chunks=kwargs["chunks"],
            model=kwargs["model"],
            top_k=kwargs.get("top_k", 3)
        )
