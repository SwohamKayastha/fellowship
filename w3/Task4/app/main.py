from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.workflow import run_workflow

app = FastAPI(title="Agentic Text-to-SQL")


class QueryPayload(BaseModel):
    question: str


@app.post("/query")
def query(payload: QueryPayload):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return run_workflow(payload.question)


@app.get("/health")
def health():
    return {"status": "ok"}
