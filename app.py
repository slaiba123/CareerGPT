from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# ── Text splitter ──────────────────────────────────────────────────────────────
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

# ── Embeddings ─────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# ── Gemini LLM ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"))

# ── Prompt Template ────────────────────────────────────────────────────────────
resume_summary_template = """
You are CareerGPT — a brutally honest AI career coach. You don't sugarcoat. You tell candidates exactly where they stand and what they need to fix.

Analyze the resume below and provide a structured breakdown with the following sections:

🎯 CAREER SNAPSHOT
One punchy paragraph summarizing who this candidate is and what they're positioning themselves for.

💪 STRENGTHS
What this resume does well. Be specific — mention actual skills, tools, or experiences that stand out.

⚠️ WEAKNESSES
What's holding this resume back. Be direct. If it's vague, underpowered, or missing something recruiters look for — say it.

🛠️ SKILLS & EXPERTISE
List the candidate's key technical and soft skills based on the resume.

📈 CAREER TRAJECTORY
Where has this person been and where are they headed? Is the progression logical and strong?

🎓 EDUCATION
Summarize their educational background and whether it supports their career goals.

🏆 NOTABLE ACHIEVEMENTS
Pull out anything impressive — numbers, awards, projects, or impact statements.

🚀 TOP 3 RECOMMENDATIONS
The 3 most important things this candidate should fix or add to their resume right now.

Be direct, specific, and useful. No filler. No generic advice.

Resume:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

resume_analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_prompt,
)


# ── QA Function ────────────────────────────────────────────────────────────────
def perform_qa(query):
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = rqa.invoke(query)
    return result['result']


# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text and build vector store
        resume_text = extract_text_from_pdf(file_path)
        splitted_text = text_splitter.split_text(resume_text)
        vectorstore = FAISS.from_texts(splitted_text, embeddings)
        vectorstore.save_local("vector_index")

        # Run resume analysis
        resume_analysis = resume_analysis_chain.invoke({"resume": resume_text})["text"]

        return render_template('results.html', resume_analysis=resume_analysis)


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query)
        return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)