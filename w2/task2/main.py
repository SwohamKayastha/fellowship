from fastapi import FastAPI
from router import router
from logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Classic Models Customer API",
    description="A layered FastAPI application for managing customer data.",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    logger.info("GET / — health check")
    return {"message": "Classic Models API is running. Visit /docs for Swagger UI."}
