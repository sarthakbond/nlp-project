import faiss
import numpy as np


def build_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    embeddings = np.ascontiguousarray(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding, top_k=3):
    query_embedding = np.asarray(query_embedding, dtype="float32")
    query_embedding = np.ascontiguousarray(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    return distances, indices