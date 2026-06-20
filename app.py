import streamlit as st
import requests
import uuid

# 1. Initialize a unique session ID for this user's browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

FASTAPI_URL = "http://localhost:8080"

st.set_page_config(page_title="Llama + Gemini RAG App", layout="centered")
st.title("📚 Llama & Gemini Flash RAG App")

# Display session ID for debugging/verification (Optional, you can remove this later)
st.caption(f"Session ID: {st.session_state.session_id[:8]}...")

tab1, tab2 = st.tabs(["📤 Upload Document", "💬 Ask Questions"])

with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            with st.spinner("Uploading and isolating your document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                # 2. Pass the session_id to the backend as form data
                data = {"session_id": st.session_state.session_id} 
                
                try:
                    response = requests.post(f"{FASTAPI_URL}/upload", files=files, data=data)
                    if response.status_code == 200:
                        st.success(response.json().get("message"))
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend.")

with tab2:
    st.header("Ask your Document")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # 3. Pass the session_id along with the query
                    payload = {"query": user_query, "session_id": st.session_state.session_id}
                    response = requests.post(f"{FASTAPI_URL}/query", json=payload)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.subheader("Answer:")
                        st.markdown(res_data["answer"])
                        with st.expander("🔍 View Context Sources Used"):
                            for idx, chunk in enumerate(res_data["context_used"], 1):
                                st.markdown(f"> **Chunk {idx}:** {chunk}")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend connection failed.")