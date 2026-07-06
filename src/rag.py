import numpy as np

from src.vector_store import search_index


def retrieve(query, index, chunks, model, top_k=3):
    """
    Retrieve the most relevant chunks for a query.
    """

    query_embedding = model.encode([query])
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    query_embedding = np.ascontiguousarray(query_embedding)

    distances, indices = search_index(
        index,
        query_embedding,
        top_k
    )

    retrieved_chunks = [chunks[idx] for idx in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    return context, retrieved_chunks