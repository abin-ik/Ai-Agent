import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Gemini Agentic RAG", layout="wide")
st.title("🤖 Agentic RAG")

# Initialize session state for chat history and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to index", 
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"]
    )
    
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Prepare files for FastAPI
                files_to_send = [
                    ("files", (file.name, file.getvalue(), file.type)) 
                    for file in uploaded_files
                ]
                
                try:
                    response = requests.post(f"{BASE_URL}/upload", files=files_to_send)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Indexed {data['total_chunks_indexed']} chunks.")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            st.warning("Please select files first.")

# --- MAIN CHAT INTERFACE ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(f"Source: {message['source']} | Tools: {', '.join(message['tools'])}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "message": prompt,
                "thread_id": st.session_state.thread_id
            }
            try:
                response = requests.post(f"{BASE_URL}/chat", json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    answer = res_data["answer"]
                    source = res_data["source_used"]
                    tools = res_data["tools_called"]

                    st.markdown(answer)
                    st.caption(f"Source: {source} | Tools: {', '.join(tools) if tools else 'None'}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "source": source,
                        "tools": tools
                    })
                else:
                    st.error("Failed to get a response from the agent.")
            except Exception as e:
                st.error(f"Error: {e}")