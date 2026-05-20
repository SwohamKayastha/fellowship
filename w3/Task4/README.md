# Task4 Agentic Text-to-SQL

This folder contains a production-ready agentic text-to-SQL application built with LangGraph. It plans, generates, validates, executes, and summarizes SQL queries over the classicmodels PostgreSQL schema.

## What it does
- Plans a query from a natural language question
- Generates a SELECT-only PostgreSQL query
- Validates the SQL to block destructive statements
- Executes the query against PostgreSQL
- Summarizes results in natural language

## Project layout
- app/agents: planner, generator, validator, executor, summarizer
- app/graph: LangGraph workflow definition
- app/prompts: centralized system prompts and schema
- app/sql/seed.sql: classicmodels schema
- app/streamlit_app.py: chat UI
- app/main.py: FastAPI entrypoint
- app/docker-compose.yml and app/Dockerfile: container setup

## ScreenShots ( Proof of Works )
![SS](https://github.com/SwohamKayastha/fellowship/blob/80d266fc06aea7d0141b35ebf6e20ff8617ccb17/w3/Task4/static/image.png)
![SS](https://github.com/SwohamKayastha/fellowship/blob/80d266fc06aea7d0141b35ebf6e20ff8617ccb17/w3/Task4/static/image2.png)
![SS of Agentic AI running SQL query](https://github.com/SwohamKayastha/fellowship/blob/80d266fc06aea7d0141b35ebf6e20ff8617ccb17/w3/Task4/static/image3.png)
![SS](https://github.com/SwohamKayastha/fellowship/blob/80d266fc06aea7d0141b35ebf6e20ff8617ccb17/w3/Task4/static/image4.png)
![SS](https://github.com/SwohamKayastha/fellowship/blob/80d266fc06aea7d0141b35ebf6e20ff8617ccb17/w3/Task4/static/image5.png)

## Configuration
Create app/.env based on app/.env.example. Required settings:
- DATABASE_URL
- LLM_BASE_URL (OpenAI compatible local endpoint)
- LLM_MODEL
- LLM_API_KEY (optional for local servers)

Example (local run):
- DATABASE_URL=postgresql://admin:password123@localhost:5433/classicmodels
- LLM_BASE_URL=http://localhost:8080/v1
- LLM_MODEL=local

Example (Docker app):
- DATABASE_URL=postgresql://postgres:postgres@db:5432/classicmodels
- LLM_BASE_URL=http://host.docker.internal:8080/v1
- LLM_MODEL=local

## Run locally (no Docker)
1) Start Postgres and load app/sql/seed.sql
2) Install deps: pip install -r app/requirements.txt
3) Run Streamlit: streamlit run app/streamlit_app.py
4) Open http://localhost:8501

## Run with Docker
1) From app/: docker compose up -d
2) Open http://localhost:8501

## API usage
If you want the API server:
- Start with: uvicorn app.main:app --reload --port 8000
- POST /query with JSON: {"question": "List employees"}

## Output format
The workflow returns a structured response:
{
  "question": "...",
  "decomposition": {
    "intent": "...",
    "tables": [],
    "columns": [],
    "filters": [],
    "joins": []
  },
  "sql": "SELECT ...;",
  "result": [ ... ],
  "summary": "Natural language answer.",
  "status": "success|failed",
  "retry_count": 0,
  "duration_ms": 0,
  "error": null
}

## Notes
- Only SELECT queries are allowed.
- Column names are camelCase and must be double-quoted.
- The LLM endpoint must be OpenAI compatible.
