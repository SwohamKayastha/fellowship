# SQL Agent (Week 3)

Text-to-SQL agent for the classicmodels PostgreSQL database.

## Project structure
```
SQL_AGENT/
	agent.py                # FastAPI service entrypoint
	benchmark.py            # Batch runner for CSV questions
	database.py             # Postgres connection + query execution
	llm.py                  # LLM calls + SQL generation + summarization
	pipeline.py             # End-to-end orchestration
	validator.py            # SQL safety checks
	benchmark_questions.csv # Sample questions
	seed.sql                # DB seed data
	docker-compose.yml      # Local Postgres container
	requirements.txt        # Python deps
	README.md               # This file
	logs/                   # Runtime logs and results
```

## What it does
- Takes a natural language question
- Decomposes intent, tables, columns, filters, joins
- Generates SELECT-only SQL
- Validates and executes against PostgreSQL
- Summarizes the result

## Prerequisites
- Docker (for PostgreSQL)
- Python 3.10+
- llama-server binary and a local model file

## Setup
1) Start PostgreSQL
```
docker compose up -d
```

2) Start the local LLM server (separate terminal) that is gemma-4 e4b model which is quantized
```
.\build\bin\Release\llama-server.exe   -m models\gemma-4-e4b-it.Q4_K_M.gguf  -ngl 99   -c 8192   --parallel 2   --cont-batching   --cache-type-k q4_0   --cache-type-v q4_0   -fa on
```

3) Install Python dependencies
```
pip install -r requirements.txt
```

4) Run the API
```
uvicorn agent:app --reload --port 8000
```

5) Test the endpoint
```
curl -X POST http://localhost:8000/agent/sql -H "Content-Type: application/json" -d "{\"question\":\"How many customers are from the USA?\"}"
```

## Configuration
Environment variables are loaded from .env in this folder.
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
- LLAMA_URL (default: http://localhost:8080/v1/chat/completions)

## ScreenShot ( Proof of Work ) 
![SS of dashboard]([w3\Task3\static\image.png](https://github.com/SwohamKayastha/fellowship/blob/main/w3/Task3/static/image.png))
![SS of SQL Query Ran of Qns 1]([w3\Task3\static\image2.png](https://github.com/SwohamKayastha/fellowship/blob/main/w3/Task3/static/image2.png))
![SS of SQL Query Ran of Qns 48]([w3\Task3\static\image3.png](https://github.com/SwohamKayastha/fellowship/blob/main/w3/Task3/static/image3.png))
![SS of SQL Query Ran of Qns 4]([w3\Task3\static\image4.png](https://github.com/SwohamKayastha/fellowship/blob/main/w3/Task3/static/image4.png))

## Benchmark
If you have a CSV of questions (column name: question), run:
```
python benchmark.py --questions benchmark_questions.csv --out logs/benchmark_results.json
```

## Important notes
- All column names are camelCase and must be double-quoted.
- The LLM sometimes returns SQL in fenced blocks; llm.py strips them.
- Increase llama-server threads or ctx size if latency is high.
