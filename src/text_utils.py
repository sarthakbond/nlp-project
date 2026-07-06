import re


def clean_text(text):
    """
    Clean extracted PDF text.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Parameters
    ----------
    text : str
        Input text.
    chunk_size : int
        Number of characters in each chunk.
    overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[str]
        List of text chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk:
            chunks.append(chunk)

        if i + chunk_size >= len(text):
            break

    return chunks