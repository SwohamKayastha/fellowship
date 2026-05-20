import logging
import os
import time

from dotenv import load_dotenv

from database import run_query
from llm import decompose, fix_sql, generate_sql, summarise
from validator import validate

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def run(question: str, max_retries: int = 1) -> dict:
    t0 = time.time()

    log.info("Decomposing: %s", question)
    decomp = decompose(question)
    log.info("Decomposition: %s", decomp)

    sql = generate_sql(question, decomp)
    log.info("Generated SQL: %s", sql)

    result = None
    error = None
    retries = 0

    for attempt in range(max_retries + 1):
        ok, reason = validate(sql)
        if not ok:
            error = f"Blocked: {reason}"
            break

        try:
            result = run_query(sql)
            log.info("Success -- %d rows (attempt %d)", len(result), attempt + 1)
            break
        except Exception as exc:
            error = str(exc)
            log.warning("Attempt %d failed: %s", attempt + 1, error)
            if attempt < max_retries:
                sql = fix_sql(sql, error)
                retries += 1
                log.info("Retrying with fixed SQL: %s", sql)

    if result is not None:
        summary = summarise(question, sql, result)
    else:
        summary = f"Could not answer. Last error: {error}"

    return {
        "question": question,
        "decomposition": decomp,
        "sql": sql,
        "result": result,
        "summary": summary,
        "status": "success" if result is not None else "failed",
        "retry_count": retries,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "error": error,
    }
