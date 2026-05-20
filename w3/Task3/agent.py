import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import run

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "agent.log")),
        logging.StreamHandler(),
    ],
)

app = FastAPI(title="Mini SQL Agent")


class QuestionPayload(BaseModel):
    question: str


@app.post("/agent/sql")
def agent(payload: QuestionPayload):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return run(payload.question, max_retries=3)


@app.get("/health")
def health():
    return {"status": "ok"}
