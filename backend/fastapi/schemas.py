from pydantic import BaseModel
import config 

class QueryRequest(BaseModel):
    query: str
    model: str = config.LLM_MODEL

class Source(BaseModel):
    text: str
    source: str
    chunk_index: int
    
class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

