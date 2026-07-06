import google.generativeai as genai


def load_model(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def generate_answer(model, question, context):
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say: "Not found in document."

Context:
{context}

Question:
{question}

Answer in a clear and concise way:
"""

    response = model.generate_content(prompt)
    return response.text