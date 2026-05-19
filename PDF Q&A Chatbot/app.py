import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from groq import Groq


# Read API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# Global vector database
vectorstore = None


def process_pdf(pdf_file):

    global vectorstore

    if pdf_file is None:
        return "Please upload a PDF."

    # Load PDF
    loader = PyPDFLoader(pdf_file.name)

    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector DB
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return "PDF processed successfully."


def ask_question(question):

    global vectorstore

    if vectorstore is None:
        return "Please process a PDF first."

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    # Groq response
    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        max_tokens=400
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:

    gr.Markdown("# PDF Q&A Chatbot")

    pdf_input = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    process_button = gr.Button("Process PDF")

    status_output = gr.Textbox(
        label="Status"
    )

    process_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status_output
    )

    question_input = gr.Textbox(
        label="Ask Question",
        placeholder="Ask anything from the PDF..."
    )

    ask_button = gr.Button("Get Answer")

    answer_output = gr.Textbox(
        lines=10,
        label="Answer"
    )

    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch()