import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths clearly
INDEX_PATH = "db/faiss_index"
INDEX_FILE = os.path.join(INDEX_PATH, "index.faiss")

# Using your installed sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_and_index(file_path: str):
    # 1. Load the document
    loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
    documents = loader.load()
    
    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    # 3. Handle the FAISS Index
    # Check if the ACTUAL index file exists, not just the directory
    if os.path.exists(INDEX_FILE):
        print(f"--- Adding to existing index at {INDEX_PATH} ---")
        db = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)
    else:
        print("--- Creating new index ---")
        db = FAISS.from_documents(chunks, embeddings)
    
    # 4. Save locally (creates the directory if it doesn't exist)
    db.save_local(INDEX_PATH)
    
    return len(chunks)