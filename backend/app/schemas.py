from typing import Optional
from pydantic import BaseModel

class CodeRequest(BaseModel):
    code: str
    language: str = "c"   # default to C for backwards compatibility
    mode: str
    user_query: str
    user_id: str
    intent: Optional[str] = None
