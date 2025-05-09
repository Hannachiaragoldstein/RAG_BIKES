import os
import getpass
import duckdb

from langchain.chat_models import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
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

# === Initialize SQL Database ===
db = SQLDatabase.from_uri("duckdb:///bike_data.db")

# === Azure OpenAI Setup ===
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure OpenAI: ")

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoai-internshi-rag.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"

llm = AzureChatOpenAI(
    model_name="gpt-4o",
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview",
    openai_api_version="2025-01-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0.0,
)

# === Load SQL Agent Prompt ===
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(
    dialect="SQLite",
    top_k=5
) + "\nAlways try to answer the question using the tables. If information is insufficient, say clearly what’s missing."

# === Create Agent Executor ===
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=60,
    early_stopping_method="generate",
    model_kwargs={"seed": 42, "temperature": 0.0},
)

# === Run Example Query ===
question = "Who is the best-selling product based on the highest total sales amount?"
answer = agent_executor.run(question)
print("Answer:", answer)
