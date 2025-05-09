Here's a complete and clean `README.md` file that documents the three Python scripts you've listed: `RAG_BIKES.py`, `RAG_BIKES_AGENTS.py`, and `RAG_BIKES_POSTGRES.py`.

---

### 📘 README.md

````markdown
# 🚲 RAG_BIKES: LangChain + Azure OpenAI + PostgreSQL Demo

This project demonstrates a **Retrieval-Augmented Generation (RAG)** workflow using LangChain, Azure OpenAI, and both **CSV-based** and **PostgreSQL-based** SQL backends. It includes three main scripts:

---

## 📁 Contents

| File | Description |
|------|-------------|
| `RAG_BIKES.py` | Basic RAG pipeline using local CSVs loaded into an in-memory SQL database. |
| `RAG_BIKES_AGENTS.py` | LangChain agent-based approach to query structured SQL data using a natural language question. |
| `RAG_BIKES_POSTGRES.py` | End-to-end Azure PostgreSQL pipeline: upload CSVs, query using LangChain + Azure OpenAI. |

---

## 🔧 Requirements

Install dependencies:

```bash
pip install pandas sqlalchemy psycopg2-binary azure-identity langchain
````

You also need:

* Azure OpenAI access
* Azure PostgreSQL server with AAD integration
* Local access to bike shop CSV files

---

## 🚀 Usage

### 1. `RAG_BIKES.py` — Simple RAG on CSVs

* Loads local CSVs into an in-memory SQL database.
* Uses LangChain to generate SQL queries from natural language.
* Uses Azure OpenAI to generate final answers from SQL results.

```bash
python RAG_BIKES.py
```

---

### 2. `RAG_BIKES_AGENTS.py` — LangChain Agent with SQL Toolkit

* Uses LangChain `SQLDatabaseToolkit` and a ReAct agent.
* Agent autonomously writes SQL queries and retrieves answers.
* Useful for more flexible natural language interaction with tabular data.

```bash
python RAG_BIKES_AGENTS.py
```

---

### 3. `RAG_BIKES_POSTGRES.py` — Azure PostgreSQL Integration

* Connects to Azure PostgreSQL using Azure Active Directory (AAD).
* Uploads bike CSV data into PostgreSQL tables.
* Builds a multi-step pipeline:

  1. Generate SQL query
  2. Execute query
  3. Generate final answer using Azure OpenAI

```bash
python RAG_BIKES_POSTGRES.py
```

---

## 🔐 Authentication & Configuration

Set the following environment variables before running:

```bash
export AZURE_OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

The PostgreSQL script uses `DefaultAzureCredential` from Azure Identity, so make sure you're logged in via `az login`.

---

## 📂 Sample CSVs (Expected)

Ensure these files exist in your `archive/` folder for uploading to the database:

```
brands.csv
categories.csv
customers.csv
orders.csv
order_items.csv
products.csv
staffs.csv
stocks.csv
stores.csv
```

---

## 📞 Example Question

```
"Who are the top 5 customers by number of orders?"
```

---

## 👤 Author

**Hanna Chiara Goldstein**
Project developed during internship at **Dataroots**.

