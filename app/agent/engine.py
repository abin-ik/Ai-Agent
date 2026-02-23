import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from app.agent.tools import tools

# 1. Load the API key from your .env file
load_dotenv()

# 2. Initialize the LLM via OpenRouter
# Note: api_key is the parameter name ChatOpenAI expects
llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite", 
    api_key=os.getenv("GOOGLE_API_KEY"), 
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# 3. Short-term memory saver
memory = MemorySaver()

# 4. Fixed instructions string
system_instr = (
    "CRITICAL INSTRUCTION:\n"
    "1. You are a RAG-FIRST assistant. For ANY query about individuals, 'experience', 'projects', or 'documents', "
    "you MUST execute 'rag_search' as your very first action.\n"
    "2. DO NOT use 'web_search' unless 'rag_search' returns 'No relevant information found'.\n"

)

# 5. Create the agent
agent_executor = create_react_agent(
    llm, 
    tools, 
    prompt=system_instr, 
    checkpointer=memory
)