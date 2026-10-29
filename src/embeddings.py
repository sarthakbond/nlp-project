import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_embeddings(chunks, max_features=5000, ngram_range=(1, 2)):
    """
    Topic #9: Fits a TF-IDF vectorizer on document chunks and produces
    sparse TF-IDF vector representations.

    Parameters
    ----------
    chunks : list[str]
        List of text chunks from the PDF.
    max_features : int
        Maximum number of vocabulary terms.
    ngram_range : tuple
        Lower and upper boundary of n-values for n-grams.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer with learned vocabulary and IDF weights.
    tfidf_matrix : scipy.sparse matrix
        Document-term matrix representing each chunk.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix


def create_dense_embeddings(chunks, model):
    """
    Generate dense vector embeddings using Sentence Transformers (Advanced Mode).
    """
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype(np.float32)
    return embeddings


# Alias for backward compatibility
create_embeddings = create_dense_embeddings
