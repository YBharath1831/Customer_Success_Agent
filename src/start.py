# import os, requests
# from dotenv import load_dotenv
# import sqlite3

# import json
# from pathlib import Path
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from prometheus_client import Counter, Histogram, start_http_server

# load_dotenv()

# PROJECT_DIR = Path(__file__).resolve().parent.parent
# DATA_FILE = PROJECT_DIR / "data" / "documentation.json"
# STORAGE_DIR = PROJECT_DIR / "storage"
# STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# API_KEY = os.environ["OPENROUTER_API_KEY"]
# MODEL = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")

# # response = requests.post(
# #     "https://openrouter.ai/api/v1/chat/completions",
# #     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
# #     json={"model": os.environ["MODEL_NAME"],
# #           "messages": [{"role": "user", "content": "Reply with the word ready."}]},
# #     timeout=30,
# # )

# # print(response.status_code)
# # print(response.json())

# with DATA_FILE.open("r", encoding="utf-8") as f:
#     documents = json.load(f)

# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
# texts, metadatas = [], []

# for doc in documents:
#     for chunk in splitter.split_text(doc["content"]):
#         texts.append(chunk)
#         metadatas.append({"doc_id": doc["doc_id"], "title": doc["title"]})

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# store = Chroma.from_texts(
#     texts=texts, metadatas=metadatas, embedding=embeddings,
#     persist_directory=str(STORAGE_DIR / "chroma"),
# )
# store.persist()
# print(f"stored {len(texts)} passages")

# connection = sqlite3.connect(STORAGE_DIR / "decisions.db")


# connection.execute("""
#     CREATE TABLE IF NOT EXISTS decisions (
#         decision_id     TEXT PRIMARY KEY,
#         created_at      TEXT NOT NULL,
#         ticket_id       TEXT NOT NULL,
#         stage           TEXT NOT NULL,
#         prediction      TEXT,
#         confidence      REAL,
#         threshold       REAL,
#         action_taken    TEXT NOT NULL,
#         reason          TEXT NOT NULL,
#         sources_used    TEXT,
#         guardrails      TEXT,
#         prompt_version  TEXT,
#         requirement_ids TEXT
#     )
# """)

# connection.commit()


# TICKETS = Counter("tickets_processed_total", "Tickets processed", ["channel", "outcome"])
# LATENCY = Histogram("response_seconds", "End to end response time")
# GUARDRAIL = Counter("guardrail_blocks_total", "Responses blocked", ["guardrail"])

# start_http_server(8001)  # metrics available at localhost:8001/metrics

# # then, in your handler
# with LATENCY.time():
#     result = process(ticket)

# TICKETS.labels(channel=ticket.channel, outcome=result.action).inc()
