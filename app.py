import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8080"

st.set_page_config(page_title="Llama + Gemini RAG App", layout="centered")
st.title("📚 Llama & Gemini Flash RAG App")
st.write("Upload clean files to S3 + Vector space, and generate validation references dynamically.")

tab1, tab2 = st.tabs(["📤 Upload Document", "💬 Ask Questions"])

# --- TAB 1: UPLOAD DOCUMENT ---
with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            with st.spinner("Uploading copy to S3 and embedding text with Llama..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{FASTAPI_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json().get("message"))
                    else:
                        st.error(f"Error from engine processing: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection failure: Is the backend engine active on port 8080?")

# --- TAB 2: ASK QUESTIONS ---
with tab2:
    st.header("Ask your Document")
    user_query = st.text_input("Enter your question based on uploaded documents:")
    
    if st.button("Get Answer"):
        if not user_query.strip():
            st.warning("Please input a valid inquiry phrase first.")
        else:
            with st.spinner("Searching Pinecone and generating answer with Gemini..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}/query", json={"query": user_query})
                    if response.status_code == 200:
                        res_data = response.json()
                        
                        st.subheader("Answer:")
                        st.markdown(res_data["answer"])
                        
                        st.write("---")
                        with st.expander("🔍 View Context Sources Used by Gemini"):
                            for idx, chunk in enumerate(res_data["context_used"], 1):
                                st.markdown(f"##### Source Chunk {idx}")
                                st.markdown(f"> {chunk}")
                    else:
                        st.error(f"Error payload generated: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Unable to securely reach API parsing framework loops.")