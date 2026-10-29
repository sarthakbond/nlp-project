import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def get_ollama_models(host=OLLAMA_BASE_URL):
    """
    Fetch all locally downloaded Ollama models from the local daemon.
    """
    try:
        res = requests.get(f"{host}/api/tags", timeout=3)
        if res.status_code == 200:
            models = res.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


class OllamaWrapper:
    """
    Local AI generation wrapper using Ollama REST API.
    """
    def __init__(self, model_name="llama3.1:latest", host=OLLAMA_BASE_URL):
        self.model_name = model_name
        self.host = host

    def generate_content(self, prompt):
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=120)
        res.raise_for_status()
        text = res.json().get("response", "")

        class ResponseWrapper:
            def __init__(self, text):
                self.text = text

        return ResponseWrapper(text)


class GeminiWrapper:
    """
    Google Generative AI wrapper supporting modern Gemini models with fallback.
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
                if "404" in err_str or "not found" in err_str or "no longer available" in err_str:
                    continue
                else:
                    raise e

        raise last_error


def load_model(engine="Ollama", ollama_model="llama3.1:latest", api_key=""):
    """
    Flexible loader for local Ollama, cloud Gemini, or Extractive QA.
    """
    if engine == "Ollama":
        return OllamaWrapper(model_name=ollama_model)
    elif engine == "Gemini" and api_key and api_key.strip():
        try:
            return GeminiWrapper(api_key=api_key)
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini model: {e}")
            return None
    return None


def extract_extractive_answer(question, context, top_sentences=2):
    """
    Topic #9: Extractive QA using TF-IDF sentence similarity.
    Extracts the most relevant sentences from the retrieved context
    without needing an LLM.
    """
    if not context or not context.strip():
        return "No relevant context found in document."

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return context[:300] + "..."

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf_sentences = vec.fit_transform(sentences)
        query_vec = vec.transform([question])
        scores = cosine_similarity(query_vec, tfidf_sentences).flatten()

        top_idx = np.argsort(scores)[::-1][:top_sentences]
        best_sentences = [sentences[i] for i in top_idx if scores[i] > 0]

        if not best_sentences:
            return sentences[0]

        return " ".join(best_sentences)
    except Exception:
        return sentences[0]


def generate_answer(model, question, context):
    """
    Generates answer using Ollama or Gemini if available,
    otherwise falls back to classical Extractive QA.
    """
    if model is None:
        extractive = extract_extractive_answer(question, context)
        return (
            f"**[Extractive QA Result - Topic #9 Classical Retrieval]:**\n\n"
            f"> {extractive}\n\n"
            f"*(Tip: Enable Ollama or enter a Gemini API Key in the sidebar for conversational synthesis.)*"
        )

    prompt = f"""You are an intelligent document QA assistant. Use ONLY the retrieved context below to answer the user's question accurately.
If the answer cannot be determined from the context, state: "The requested information is not found in the provided document."

Retrieved Context:
{context}

Question:
{question}

Provide a clear, accurate, and concise answer based strictly on the context above:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        fallback = extract_extractive_answer(question, context)
        return (
            f"**[Model Generation Error: {e}]**\n\n"
            f"**Fallback Extractive Answer:**\n\n> {fallback}"
        )
