from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default_user" # Added for history tracking

class ChatResponse(BaseModel):
    answer: str
    source_used: str
    tools_called: List[str]