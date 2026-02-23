# CareerGPT 🚀
### *"Your resume has been lying to you. CareerGPT won't."*

An AI-powered career coaching web app that analyzes your resume, gives brutally honest feedback, and lets you ask career questions — all in one place.

---

## What It Does

- 📋 **Resume Analysis** — Upload your PDF resume and get a structured AI breakdown including strengths, weaknesses, career trajectory, and top recommendations
- 💬 **Career Q&A** — Ask anything about your resume or career and get personalized AI answers
- 🔍 **Semantic Search** — Uses vector embeddings to find the most relevant parts of your resume when answering questions
- ⚡ **Fast** — Powered by Groq's inference engine for near-instant responses

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Orchestration | LangChain |
| PDF Parsing | PyPDF2 |
| Frontend | HTML, CSS, Tailwind |
| Deployment | Render |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/slaiba123/CareerGPT.git
cd CareerGPT
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## How It Works

```
User uploads PDF resume
        ↓
PyPDF2 extracts text
        ↓
Text split into chunks → FAISS vector store (for Q&A)
        ↓
Full resume sent to Groq LLM → Structured analysis returned
        ↓
User can ask follow-up questions → FAISS retrieves relevant chunks → LLM answers
```

---

## Project Structure

```
CareerGPT/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── Procfile                # For Render deployment
├── .env                    # API keys (not committed)
├── .gitignore
├── uploads/                # Temporarily stores uploaded PDFs
├── vector_index/           # FAISS index (generated at runtime)
└── templates/
    ├── index.html          # Landing page
    ├── results.html        # Resume analysis results
    ├── ask.html            # Career Q&A input page
    └── qa_results.html     # Q&A results page
```

---

## Deployment

This app is deployed on **Render**. To deploy your own instance:

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set Start Command: `gunicorn app:app`
5. Add environment variable: `GROQ_API_KEY`
6. Deploy ✅

> **Note:** Render's free tier has an ephemeral filesystem — uploaded files and the FAISS index reset on each redeploy. This is fine for demos.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

<p align="center">Built with 🔥 by <a href="https://github.com/slaiba123">slaiba123</a></p>
