from fastapi import FastAPI ,APIRouter, Query
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from utils.embedder import get_openai_text_embedding

app = FastAPI(title="TDS Virtual TA",
    description="Ask questions about the Tools in Data Science course",
    version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
CHROMA_DIR = "./tds_chroma_db"
# Load persistent ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("tds_multimodal")

router = APIRouter()

class AskResponse(BaseModel):
    answer: str
    links: list[dict]

# Input schema


class Query(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask_virtual_ta(query: Query):
    embedding = get_openai_text_embedding(query.question)
    if embedding is None:
        return {"error": "Failed to embed query"}
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=query.top_k
    )
    responses = [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {"answers": responses}
