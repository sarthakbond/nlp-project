import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def search_tfidf(vectorizer, tfidf_matrix, query, top_k=3):
    """
    Topic #9: Question Processing & Information Retrieval via Cosine Similarity.
    
    1. Transforms query using fitted TF-IDF vectorizer.
    2. Calculates Cosine Similarity between query vector and chunk vectors.
    3. Returns top-k matching indices and their similarity scores.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    tfidf_matrix : scipy.sparse matrix
        TF-IDF representations of document chunks.
    query : str
        Natural language question asked by the user.
    top_k : int
        Number of top relevant chunks to retrieve.

    Returns
    -------
    top_scores : np.ndarray
        Cosine similarity scores for top chunks (0.0 to 1.0).
    top_indices : np.ndarray
        Indices of top chunks.
    """
    if not query or not query.strip():
        return np.array([]), np.array([])

    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of highest similarity scores
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_indices = sorted_indices[:top_k]
    top_scores = similarity_scores[top_indices]

    return top_scores, top_indices


def build_index(embeddings):
    """
    Builds a FAISS index from dense vectors (Advanced Mode).
    """
    try:
        import faiss
        embeddings = np.asarray(embeddings, dtype="float32")
        embeddings = np.ascontiguousarray(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except ImportError:
        raise ImportError(
            "faiss-cpu is not installed. Use TF-IDF mode or install faiss-cpu."
        )


def search_index(index, query_embedding, top_k=3):
    """
    Searches FAISS index using L2 distance (Advanced Mode).
    """
    query_embedding = np.asarray(query_embedding, dtype="float32")
    query_embedding = np.ascontiguousarray(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices
