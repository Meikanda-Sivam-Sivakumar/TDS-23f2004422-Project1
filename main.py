from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import base64
from PIL import Image
from io import BytesIO
import chromadb

from utils.embedder import get_openai_text_embedding, get_clip_image_embedding
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TDS Virtual TA",
    description="Ask questions about the Tools in Data Science course",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ChromaDB
CHROMA_DIR = "./tds_chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("tds_multimodal_context")

# Request schema
class AskRequest(BaseModel):
    question: str
    attachments: Optional[List[str]] = None  # base64-encoded images

# Response schema
class Link(BaseModel):
    url: str
    text: str

class AskResponse(BaseModel):
    answer: str
    links: List[Link]

@app.post("/api/", response_model=AskResponse)
def ask_virtual_ta(req: AskRequest):
    text_embedding = get_openai_text_embedding(req.question)
    if text_embedding is None:
        return {"answer": "Failed to process your question", "links": []}

    image_embeddings = []
    if req.attachments:
        for b64 in req.attachments:
            try:
                img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                img_emb = get_clip_image_embedding(img)
                if img_emb is not None:
                    image_embeddings.append(img_emb)
            except Exception as e:
                print("⚠️ Image decode error:", e)

    # Combine text + image embeddings
    all_embeddings = [text_embedding] + image_embeddings if image_embeddings else [text_embedding]
    combined_embedding = np.mean(np.array(all_embeddings), axis=0).tolist()

    results = collection.query(query_embeddings=[combined_embedding], n_results=5)

    links = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        link_text = doc[:150] + "..." if len(doc) > 150 else doc
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{meta['topic'].replace(' ', '-')}/{meta.get('post_id', '')}"
        links.append({"url": url, "text": link_text})

    best_answer = results["documents"][0][0] if results["documents"][0] else "No answer found."
    return {"answer": best_answer, "links": links}
