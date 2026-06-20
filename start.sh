#!/bin/bash
# Boot the internal FastAPI backend in the background
uvicorn main:app --host 127.0.0.1 --port 8080 &

# Boot the public Streamlit app in the foreground
streamlit run app.py --server.port=8501 --server.address=0.0.0.0