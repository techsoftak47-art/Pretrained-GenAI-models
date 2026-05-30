import gradio as gr
import fitz
import tempfile

from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =====================================================
# PDF EXTRACTION
# =====================================================

def extract_pdf_text(pdf_file):

    text = ""

    doc = fitz.open(pdf_file.name)

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


# =====================================================
# CHUNKING
# =====================================================

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# =====================================================
# EMBEDDINGS
# =====================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# VECTOR STORE
# =====================================================

def build_vector_store(chunks):

    vector_db = FAISS.from_texts(
        chunks,
        embedding_model
    )

    return vector_db


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_context(vector_db, query):

    docs = vector_db.similarity_search(
        query,
        k=15
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context


# =====================================================
# GROQ ANALYSIS
# =====================================================

def generate_report(groq_key, context):

    client = Groq(api_key=groq_key)

    prompt = f"""
You are an expert academic research analyst.

Analyze the uploaded research papers.

Context:
{context}

Generate a detailed report containing:

1. Executive Summary

2. Research Themes

3. Methodologies Used
   - Algorithms
   - Models
   - Datasets
   - Evaluation Metrics

4. Author Mentioned Limitations

5. Research Gaps

6. Underexplored Areas

7. Future Research Opportunities

8. Comparative Analysis

9. 10 Novel Research Ideas

Return a well-structured report.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content":
                "You are an expert research assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )

    return response.choices[0].message.content


# =====================================================
# MAIN FUNCTION
# =====================================================

def analyze_papers(pdf_files, groq_api_key):

    if not pdf_files:
        return "Please upload at least one PDF."

    if not groq_api_key:
        return "Please enter Groq API Key."

    all_text = ""

    for pdf in pdf_files:

        paper_text = extract_pdf_text(pdf)

        all_text += (
            f"\n\n===== PAPER =====\n\n"
            + paper_text
        )

    chunks = create_chunks(all_text)

    vector_db = build_vector_store(chunks)

    context = retrieve_context(
        vector_db,
        "research themes methodologies limitations "
        "research gaps future work"
    )

    report = generate_report(
        groq_api_key,
        context
    )

    return report


# =====================================================
# GRADIO UI
# =====================================================

with gr.Blocks(
    title="Research Gap Analyzer"
) as demo:

    gr.Markdown(
        """
# 📚 Generative AI Research Gap Analyzer

Upload multiple research papers and automatically discover:

✅ Research Themes

✅ Methodologies

✅ Limitations

✅ Research Gaps

✅ Future Research Opportunities

Powered by:
- FAISS
- LangChain
- Groq
- Llama-3.3-70B-Versatile
        """
    )

    pdf_input = gr.File(
        file_count="multiple",
        file_types=[".pdf"],
        label="Upload Research Papers"
    )

    groq_key = gr.Textbox(
        label="Groq API Key",
        type="password"
    )

    analyze_btn = gr.Button(
        "Analyze Papers"
    )

    output = gr.Markdown(
        label="Analysis Report"
    )

    analyze_btn.click(
        fn=analyze_papers,
        inputs=[
            pdf_input,
            groq_key
        ],
        outputs=output
    )

demo.launch()