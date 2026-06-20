#!/bin/bash
# Start FastAPI backend engine internally in background context loops
uvicorn main:app --host 127.0.0.1 --port 8080 &

# Start public Streamlit engine routing execution systems
streamlit run app.py --server.port=8501 --server.address=0.0.0.0