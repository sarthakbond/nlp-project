import numpy as np

def create_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype(np.float32)
    return embeddings