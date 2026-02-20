import os
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from rag1.ingestion import embeddings, INDEX_PATH

@tool
def rag_search(query: str):
    """Search this for private docs or company files."""
    if not os.path.exists(INDEX_PATH):
        return "Knowledge base is empty."
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

@tool
def web_search(query: str):
    """Search this for general news, weather, or public info."""
    search = DuckDuckGoSearchRun()
    return search.run(query)
tools = [rag_search, web_search]    