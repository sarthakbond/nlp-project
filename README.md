# ⚖️ Advocium · LexiQA (Legal & Compliance Intelligence QA Engine)

> **Keystone School of Engineering · Department of Artificial Intelligence & Data Science**  
> **Topic #9 Curriculum Alignment: Question Answering System (TF-IDF + Cosine Similarity)**  
> **Team:** Skill G &middot; **Members:** Sarthak Phadatare, Pranit Latpate, Aqib Pansare, Shamita Marathe

---

## 🎯 Real-World Problem: Contract Review & Compliance Intelligence

Legal teams, compliance auditors, contract managers, and students routinely face dense 40+ page documents (such as **Mutual NDAs, Master Services Agreements, Executive Employment Contracts, and GDPR Data Processing Addenda**). Manually finding critical terms (indemnification caps, termination windows, confidentiality survivability, and regulatory breach notice deadlines) is slow and error-prone.

**LexiQA** solves this real-world bottleneck by implementing **Topic #9: Question Answering System** as an automated legal clause retrieval and extractive question answering engine.

---

## 🔬 Curriculum Topic #9 Alignment

| Rubric Dimension | Syllabus Requirement | LexiQA Implementation |
| :--- | :--- | :--- |
| **System Category** | Question Answering System | Legal & Compliance Intelligence QA Engine |
| **Objective** | Provide answers to questions based on given document | Ingests legal agreements, extracts exact clauses, and generates grounded answers |
| **Methodology** | Question Processing, Information Retrieval | Lowercasing, punctuation stripping, n-grams, TF-IDF vectorization, Cosine ranking |
| **Mandated Algorithm** | **TF-IDF + Cosine Similarity** | Implemented in **Python `scikit-learn`** and **in-browser Web NLP Engine** |
| **Deployment Mode** | Cloud-ready & Zero Backend Dependency | **Single Vercel deployment** with automatic client-side fallback |

---

## 📐 Mathematical Formulation

### 1. Term Frequency – Inverse Document Frequency (TF-IDF)
For each term $t$ in a contractual clause $d$ across a corpus of clauses $D$:

$$\text{TF}(t, d) = 1 + \ln(f_{t, d})$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2. Cosine Similarity for Information Retrieval
Given the vectorized query $\mathbf{q}$ and document clause vector $\mathbf{d}$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

Clauses are ranked descending by Cosine Similarity, and the top-$k$ most relevant passages are selected for clause extraction and legal risk analysis.

---

## 🚀 Key Features

* **Dual-Pipeline Zero-Failure Engine:**
  * **Primary:** Python serverless endpoint (`/api/query`) utilizing `scikit-learn` and `numpy`.
  * **Fallback:** High-precision in-browser Web NLP engine running directly in JavaScript. If the backend experiences cold starts or network timeouts, the client engine takes over within 50ms with zero disruption.
* **Preloaded Legal Knowledge Bases:**
  1. *Mutual Non-Disclosure Agreement (NDA)*
  2. *Enterprise Master Services Agreement (MSA) & SLA*
  3. *Executive Employment & IP Assignment Agreement*
  4. *Enterprise Data Processing Addendum (GDPR DPA)*
  5. *Custom Contract / PDF Drag & Drop Uploader*
* **Real-World Clause Risk Assessment:**
  * Identifies high-risk legal clauses (uncapped liability, indemnification, liquidated damages, immediate termination, non-compete covenants).
* **Topic #9 Mathematical Breakdown Inspector:**
  * Displays active vocabulary tokens, query-document term overlap, vector norms, and exact Cosine Similarity decimals in real time.
* **Advocium Editorial Legal Design:**
  * Premium, responsive interface optimized for both mobile devices (phones/tablets) and desktop displays.

---

## 👥 Project Team (Skill G)

* **Sarthak Phadatare**
* **Pranit Latpate**
* **Aqib Pansare**
* **Shamita Marathe**

---

## 💻 Running Locally

### 1. Start Frontend (React + Vite)
```powershell
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000).

### 2. (Optional) Run Python FastAPI Backend
```powershell
pip install -r api/requirements.txt
python -m uvicorn api.index:app --port 8000
```

*(Note: The React frontend operates seamlessly even without starting the Python server, thanks to the integrated in-browser client NLP engine fallback!)*

---

## ☁️ 1-Click Vercel Deployment

Both the frontend and backend are configured to deploy together on **Vercel** with **no external server needed (no Render required)**:

1. Push this repository to GitHub.
2. In the [Vercel Dashboard](https://vercel.com), click **Add New Project** and select this repository.
3. Vercel automatically detects [`vercel.json`](file:///c:/Users/Mohammad%20Aqib/OneDrive/Desktop/NLP%20Mini%20Project/vercel.json):
   * Builds the React frontend to `/dist`
   * Deploys `api/index.py` as a Python Serverless Function (`@vercel/python`)
4. Click **Deploy** &mdash; your full-stack legal QA application is live!
