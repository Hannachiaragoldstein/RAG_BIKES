import os
import duckdb
import getpass
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub

# === Load CSVs into DuckDB ===
csv_folder = "/Users/hannachiaragoldstein/Downloads/archive"
db_path = "bike_data.db"
con = duckdb.connect(db_path)

for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        table_name = filename.replace(".csv", "")
        file_path = os.path.join(csv_folder, filename)
        print(f"Loading {file_path} into table {table_name}...")
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path}');
        """)

con.close()
print("✅ All tables loaded into bike_data.db!")

# === Connect to SQL database ===
db = SQLDatabase.from_uri("duckdb:///bike_data.db")
print("Dialect:", db.dialect)
print("Usable Tables:", db.get_usable_table_names())
print(db.run("SELECT * FROM brands LIMIT 10;"))

# === Environment Variables for Azure OpenAI ===
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure OpenAI: ")

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoai-internshi-rag.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"

# === Initialize AzureChatOpenAI ===
llm = AzureChatOpenAI(
    model_name="gpt-4o",
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview",
    openai_api_version="2025-01-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=None
)

# === Load SQL prompt template ===
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
assert len(query_prompt_template.messages) == 2
for message in query_prompt_template.messages:
    message.pretty_print()

# === State & TypedDict Definitions ===
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# === Query Generation ===
def write_query(state: dict) -> QueryOutput:
    prompt = query_prompt_template.invoke({
        "dialect": "SQL",
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"]
    })

    prompt_str = prompt[0]['content'] if isinstance(prompt, list) else str(prompt)
    response = llm.predict(prompt_str)

    return QueryOutput(query=response.strip())

# === SQL Execution ===
def execute_query(state: State):
    tool = QuerySQLDatabaseTool(db=db)
    return {"result": tool.invoke(state["query"])}

# === Answer Generation ===
def generate_answer(state: State):
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# === Pipeline Function ===
def get_answer_for_question(state: dict):
    query_output = write_query(state)
    query_result = execute_query({"query": query_output["query"]})
    
    intermediate_state = {
        "question": state["question"],
        "query": query_output["query"],
        "result": query_result["result"],
    }

    answer_output = generate_answer(intermediate_state)
    return answer_output["answer"]

# === Example Query ===
if __name__ == "__main__":
    question = "What customer purchased the most?"
    answer = get_answer_for_question({"question": question})
    print("Answer:", answer)
