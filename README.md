# 📚 Multi-Tenant Llama & Gemini Flash RAG Application

A secure, production-grade Retrieval-Augmented Generation (RAG) pipeline built with a **FastAPI** backend and a **Streamlit** frontend interface. The application processes text and PDF documents, generates dense vector embeddings using **Llama Inference**, stores them securely in isolated vector namespaces using **Pinecone**, and answers user queries with **Gemini 2.5 Flash**.

---

## 🏗️ Architecture Overview

The project is structured modularly to isolate internal ingestion logic, database routing, and LLM inference. 

```text
RAG_PROJECT/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD automated test & AWS deployment pipeline
├── services/
│   ├── __init__.py
│   ├── config.py               # Global environment and hyperparameter configurations
│   ├── doc_parser.py           # Text extraction and advanced regex data cleaning
│   ├── google_ai.py            # Gemini 2.5 Flash context injection & generation
│   └── pinecone_db.py          # Llama text-embed-v2 vector indexing & querying
├── tests/
│   ├── __init__.py
│   ├── test_integration.py     # FastAPI endpoint verification via mock HTTP requests
│   └── test_unit.py            # Isolated internal logic and utility function testing
├── app.py                      # Streamlit frontend user interface
├── main.py                     # FastAPI core backend service gateway
├── requirements.txt            # Project execution dependencies
└── README.md                   # System documentation

🔒 Multi-Tenant Data Tenancy & Security Isolation
To prevent cross-tenant data leaks (ensuring users can never view or query documents uploaded by others), the app implements an end-to-end logical isolation strategy driven by a cryptographically secure browser-tab tracking identifier (session_id):

Frontend Tagging: Upon initialization, app.py instantiates a unique UUIDv4 token inside st.session_state. This token remains assigned to the user session for all API interactions.

Cloud Storage Isolation (S3): Documents backed up to Amazon S3 are isolated into dynamically mapped folders: uploaded-documents/{session_id}/{filename}.

Vector Database Isolation (Pinecone Namespaces): Dense vector records are partitioned using Pinecone's Namespaces capability. When chunked documents are saved, they are locked within that session's namespace. When querying, searches are strictly limited to that explicit namespace, making it technically impossible for a user query to fetch context chunks from another user.


3. Run the Backend & Frontend Simultaneously
You will need two terminal windows open with your virtual environment active in both.

Terminal 1 (FastAPI Backend Server):

Bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
Terminal 2 (Streamlit UI App):

Bash
streamlit run app.py
The application will automatically launch in your browser at http://localhost:8501.

🚀 CI/CD Infrastructure Pipeline (GitHub Actions)
The .github/workflows/deploy.yml pipeline automates standard integration testing and single-container continuous deployment to an Amazon EC2 cluster.

Pipeline Execution Workflow
Runner Initialization: Code executes inside a clean cloud-hosted Ubuntu environment.

Testing Gatekeeper: Python dependencies and testing libraries (pytest, httpx, pytest-asyncio) install, and the core test matrix runs. If a single test fails, the process aborts to prevent broken changes from hitting production.

Docker Compilation: The source code packages into a Docker image using the project's multi-port exposed configuration.

AWS Private Registry Push: The authenticated build engine pushes the newly tagged image to an Amazon Elastic Container Registry (ECR) bucket.

Secure SSH Orchestration: The workflow securely connects to the designated EC2 instance using an SSH key, stops the legacy container, updates the runtime engine, and mounts the live image across exposed host ports 8080 and 8501.