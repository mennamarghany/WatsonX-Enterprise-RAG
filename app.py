import os
import gradio as gr
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.credentials import Credentials

# ======================
# Watsonx config
# ======================
credentials = Credentials(
    api_key=os.environ["WATSONX_APIKEY"],
    url="https://eu-de.ml.cloud.ibm.com"
)

llm = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    credentials=credentials,
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={
        "max_new_tokens": 512,
        "temperature": 0.2
    }
)

# ======================
# Session memory
# ======================
sessions = {}

# ======================
# RAG chain builder
# ======================
def build_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user’s question using the document content."),
        ("human", "{input}")
    ])

    chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=prompt
    )
    return chain

# ======================
# Chat streaming function
# ======================
def chat(pdf, question, history: List[Dict[str, str]]):
    session_id = id(history)
    if session_id not in sessions:
        sessions[session_id] = build_chain(pdf.name)

    chain = sessions[session_id]

    # Streaming generator
    response_text = ""
    # Note: Watsonx ModelInference does not natively stream tokens
    # We'll simulate token streaming by splitting the final answer
    result = chain.invoke({"input": question})
    full_answer = result["answer"]

    for token in full_answer.split():
        response_text += token + " "
        yield history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": response_text.strip()}]

# ======================
# UI
# ======================
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
            question = gr.Textbox(label="Ask a question")
            chatbox = gr.Chatbot(label="Chat with your PDF")
            submit = gr.Button("Ask")
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            max_tokens = gr.Slider(1, 2048, value=512, step=1, label="Max tokens")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
    
    submit.click(
        chat,
        inputs=[pdf_file, question, chatbox],
        outputs=[chatbox]
    )

demo.launch(share=True)
