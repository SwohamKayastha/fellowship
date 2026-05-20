# Week 3 Workspace Overview

This folder contains all Week 3 artifacts: baseline and upgraded text-to-SQL systems, agentic workflow, datasets, and task-specific assets. Each subfolder corresponds to a task or experiment completed during Week 3.

## Folder summary
- SQL_AGENT: Baseline text-to-SQL FastAPI service with prompt chaining, SQL validation, and benchmark runner. Includes database seed, logs, and screenshots.
- Task3: Prompt-chaining Text-to-SQL system with Streamlit UI, evaluation script, Docker setup, and logging. Built as the non-agentic pipeline deliverable.
- Task4: Agentic Text-to-SQL system built with LangGraph and a Streamlit UI. Includes agents for planning, SQL generation, validation, execution, and summarization.
- Task1: SQL seed files used for initial task setup.
- Task2: Decomposition assignment reference (PDF).
- Task: Shared assignment PDFs and base dataset files used across Week 3 tasks.

## Structure highlights
SQL_AGENT
- agent.py: FastAPI entrypoint
- pipeline.py: End-to-end orchestration
- llm.py: LLM calls and summarization
- validator.py: SQL safety checks
- benchmark.py: Batch runner for questions

Task3
- streamlit_app.py: UI for text-to-SQL interaction
- sql_generator.py / executor.py: prompt chaining pipeline
- evaluate.py: benchmark evaluation
- Dockerfile / docker-compose.yml: local services

Task4
- app/agents: planner, generator, validator, executor, summarizer
- app/graph: LangGraph workflow
- app/prompts: centralized prompts and schema
- app/streamlit_app.py: agentic UI

## Tasks completed
- Task3: Non-agentic prompt-chaining pipeline (Text-to-SQL) with evaluation and UI.
- Task4: Agentic LangGraph workflow with planning, validation, execution, and summarization.
- SQL_AGENT: Baseline FastAPI agent with benchmark runner and logs.
