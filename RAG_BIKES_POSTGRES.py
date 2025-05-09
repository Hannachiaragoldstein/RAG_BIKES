# === Install required packages ===
# Run these in terminal (not in script):
# pip install pandas sqlalchemy psycopg2-binary azure-identity langchain

import os
import getpass
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine
from azure.identity import DefaultAzureCredential
from langchain.chat_models import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from typing_extensions import TypedDict, Annotated

# === PostgreSQL Connection Setup ===
def get_postgres_uri() -> str:
    dbhost = "aoai-internship-postgres-server.postgres.database.azure.com"
    dbname = "postgres"
    dbuser = urllib.parse.quote("hannachiara.goldstein@datarootsio.onmicrosoft.com")  # URL-encode username
    sslmode = "require"

    # Azure AD token
    credential = DefaultAzureCredential()
    password = credential.get_token("https://ossrdbms-aad.database.windows.net/.default").token

    return f"postgresql://{dbuser}:{password}@{dbhost}:5432/{dbname}?sslmode={sslmode}"

# === Upload CSVs to PostgreSQL ===
def upload_csvs_to_postgres(folder: str, csv_files: list[str]):
    engine = create_engine(get_postgres_uri())
    for csv_file in csv_files:
        table_name = csv_file.replace('.csv', '')
        file_path = os.path.join(folder, csv_file)
        print(f"Uploading {csv_file} as table '{table_name}'...")
        df = pd.read_csv(file_path)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
    print("✅ All tables uploaded successfully!")

# === Azure OpenAI Setup ===
def setup_azure_openai() -> AzureChatOpenAI:
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure OpenAI: ")

    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoai-internshi-rag.openai.azure.com/"
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"

    return AzureChatOpenAI(
        model_name="gpt-4o",
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2025-01-01-preview",
        temperature=0,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    )

# === LangChain Pipeline ===
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(llm, db, state: dict) -> QueryOutput:
    prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    prompt = prompt_template.invoke({
        "dialect": "PostgreSQL",
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"]
    })

    prompt_str = prompt[0]['content'] if isinstance(prompt, list) else str(prompt)
    response = llm.predict(prompt_str)
    return QueryOutput(query=response.strip())

def execute_query(db, query: str) -> dict:
    tool = QuerySQLDatabaseTool(db=db)
    return {"result": tool.invoke(query)}

def generate_answer(llm, state: dict) -> dict:
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def get_answer_for_question(llm, db, question: str) -> str:
    state = {"question": question}
    query_output = write_query(llm, db, state)
    query_result = execute_query(db, query_output["query"])

    full_state = {
        **state,
        "query": query_output["query"],
        "result": query_result["result"]
    }

    answer_output = generate_answer(llm, full_state)
    return answer_output["answer"]

# === Main Run Block ===
if __name__ == "__main__":
    # 1. Upload CSVs (run once, then comment out)
    csv_folder = "/home/azureuser/archive/"
    csv_files = [
        "brands.csv", "categories.csv", "customers.csv", "orders.csv",
        "order_items.csv", "products.csv", "staffs.csv", "stocks.csv", "stores.csv"
    ]
    # upload_csvs_to_postgres(csv_folder, csv_files)  # Uncomment to run once

    # 2. Set up LLM and DB connection
    llm = setup_azure_openai()
    engine = create_engine(get_postgres_uri())
    db = SQLDatabase(engine=engine)

    # 3. Ask your question
    question = "Who are the top 5 customers by number of orders?"
    answer = get_answer_for_question(llm, db, question)
    print("Answer:", answer)
