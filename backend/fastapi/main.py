from fastapi import FastAPI
import schemas 
import rag

app = FastAPI()

@app.post("/query", response_model=schemas.QueryResponse)
def query(request: schemas.QueryRequest):
    answer, sources = rag.answer_query(request.query, request.model)
    return schemas.QueryResponse(answer=answer, sources=sources)

@app.get("/health")
def health():
    return {"status": "ok"}