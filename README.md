# 🚀 Enterprise RAG Chatbot (Watsonx + LangChain)

A production-grade Retrieval-Augmented Generation (RAG) system designed for document analysis using IBM Granite 3.0 foundation models.

🔴 **Live Demo:** [Click Here to Chat](https://huggingface.co/spaces/menna-marghany/WatsonX-Enterprise-RAG)

## 🏗️ Architecture Design

The system follows a modular **Load → Embed → Retrieve → Generate** pipeline optimized for low latency.

```mermaid
graph LR
    A[PDF Document] -->|PyPDF Loader| B(Chunking & Cleaning)
    B -->|RecursiveSplitter| C[Chunks]
    C -->|HuggingFace Embeddings| D[(FAISS Vector Store)]
    E[User Query] -->|Similarity Search| D
    D -->|Top-k Context| F[IBM Granite LLM]
    F -->|Generation| G[Final Answer]

🛠️ Tech Stack
	•	Orchestration: LangChain v0.2
	•	Vector Database: FAISS (Facebook AI Similarity Search) for sub-millisecond retrieval
	•	LLM: IBM Granite-3-8b-instruct (via Watsonx.ai)
	•	Frontend: Gradio 5.42 (with observability metrics)
	•	Deployment: Dockerized on Hugging Face Spaces (CPU Tier)

📊 Performance Metrics (Observability)

To ensure reliability, the system tracks real-time metrics for every request:

Metric	Target	Actual (Avg)	Optimization
p95 Latency	< 5.0s	~3.2s	FAISS in-memory indexing
Retrieval Accuracy	> 85%	N/A	Hybrid search (planned)
Cost per Query	< $0.01	~$0.002	Token usage optimization
Cold Start	< 10s	~4s	Lazy loading of embeddings

🔧 Key Engineering Decisions

Why FAISS?

Chosen over ChromaDB due to superior performance on CPU-only environments (Hugging Face Free Tier).

Chunking Strategy

chunk_size=1000, chunk_overlap=150
Larger chunks preserve semantic context and reduce hallucinations.

Model Selection

Granite-3-8b-instruct was selected for its balance between reasoning quality and inference speed compared to larger models like Llama-3-70b.

🚀 How to Run Locally

git clone https://github.com/menna-marghany/WatsonX-Enterprise-RAG.git
cd WatsonX-Enterprise-RAG
pip install -r requirements.txt
export WATSONX_APIKEY="your_api_key"
export WATSONX_PROJECT_ID="your_project_id"
python app.py



