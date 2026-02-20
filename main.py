import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File
from app.schemas import ChatRequest, ChatResponse
from app.agent.engine import agent_executor
from rag1.ingestion import process_and_index

app = FastAPI(title="Gemini Agentic RAG")
UPLOAD_DIR = "temp_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    total_chunks = 0
    for file in files:
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        total_chunks += process_and_index(temp_path)
    return {"status": "success", "total_chunks_indexed": total_chunks}

@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [("user", request.message)]}
    
    result = await agent_executor.ainvoke(inputs, config=config)
    
    # 1. Get the last message from the result
    final_message = result["messages"][-1]
    
    # 2. SAFELY EXTRACT STRING CONTENT
    # Sometimes Gemini returns a list of parts; we need to join them into one string.
    if isinstance(final_message.content, list):
        # Extract text from each part if content is a list
        final_answer = " ".join([part.get("text", "") if isinstance(part, dict) else str(part) 
                                for part in final_message.content])
    else:
        final_answer = str(final_message.content)

    # 3. Identify tools called for the response
    tools_used = [m.name for m in result["messages"] if hasattr(m, 'name') and m.type == 'tool']
    source = "Knowledge Base" if "rag_search" in tools_used else "Gemini/Web"

    return ChatResponse(
        answer=final_answer, # Now guaranteed to be a string
        source_used=source,
        tools_called=tools_used
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)