# CSV RAG Analyst

Local RAG-based tool for querying CSV data using natural language. It uses a local LLM (via Ollama) to generate and execute pandas code from user queries.

## Overview

- Natural language queries over CSV data (EN/DE)
- Generates and runs pandas code
- Supports filtering, aggregation, grouping, and time-based queries
- Runs locally (no external APIs)
- Optional Streamlit interface

## Setup

```bash
git clone https://github.com/maralthesage/csv-rag-analyst.git
cd csv-rag-analyst

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

```bash
ollama pull gemma3:27b
streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

## Usage

* Set your CSV path in `rag_engine/loader.py`
* Ensure column names match your queries
* Ask questions through the UI

## Notes

* Runs fully locally
* Output quality depends on data quality and model choice

```
```
