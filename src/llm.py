import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GeminiWrapper:
    """
    Wrapper around Google Generative AI that supports modern Gemini models
    with automatic fallback (gemini-3.8-flash -> gemini-2.0-flash -> gemini-1.5-flash).
    """
    def __init__(self, api_key):
        self.api_key = api_key.strip()
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.candidate_models = [
            "gemini-3.8-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

    def generate_content(self, prompt):
        import google.generativeai as genai
        last_error = None

        for model_name in self.candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # If model is 404, deprecated, or unavailable, fallback to next model
                if "404" in err_str or "not found" in err_str or "no longer available" in err_str:
                    continue
                else:
                    raise e

        raise last_error


def load_model(api_key):
    """
    Initializes the Gemini model wrapper if an API key is provided.
    Returns None if api_key is missing or invalid.
    """
    if not api_key or not api_key.strip():
        return None

    try:
        return GeminiWrapper(api_key)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini model: {e}")
        return None


def extract_extractive_answer(question, context, top_sentences=2):
    """
    Topic #9: Extractive QA using TF-IDF sentence similarity.
    Extracts the most relevant sentences from the retrieved context
    without needing an external LLM / API key.
    """
    if not context or not context.strip():
        return "No relevant context found in document."

    # Split context into sentences
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return context[:300] + "..."

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf_sentences = vec.fit_transform(sentences)
        query_vec = vec.transform([question])
        scores = cosine_similarity(query_vec, tfidf_sentences).flatten()

        import numpy as np
        top_idx = np.argsort(scores)[::-1][:top_sentences]
        best_sentences = [sentences[i] for i in top_idx if scores[i] > 0]

        if not best_sentences:
            return sentences[0]

        return " ".join(best_sentences)
    except Exception:
        return sentences[0]


def generate_answer(model, question, context):
    """
    Generates answer using Gemini LLM if available,
    otherwise falls back gracefully to classical Extractive QA.
    """
    if model is None:
        extractive = extract_extractive_answer(question, context)
        return (
            f"**[Extractive QA Result - Topic #9 Classical Retrieval]:**\n\n"
            f"> {extractive}\n\n"
            f"*(Tip: Provide a `GEMINI_API_KEY` in the sidebar or `.env` file to generate a synthesized conversational response.)*"
        )

    prompt = f"""
You are an intelligent document QA assistant. Use ONLY the retrieved context below to answer the user's question accurately.
If the answer cannot be determined from the context, state: "The requested information is not found in the provided document."

Retrieved Context:
{context}

Question:
{question}

Provide a clear, accurate, and concise answer based strictly on the context above:
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback to extractive answer if API fails (e.g., quota or network issue)
        fallback = extract_extractive_answer(question, context)
        return (
            f"**[LLM API Error: {e}]**\n\n"
            f"**Fallback Extractive Answer:**\n\n> {fallback}"
        )
