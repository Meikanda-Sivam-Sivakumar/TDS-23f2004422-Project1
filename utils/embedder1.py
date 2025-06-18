from openai import OpenAI
import os
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"), base_url="https://aipipe.org/openai/v1")

def get_openai_text_embedding(text: str, max_chars: int = 7000):
    if len(text) > max_chars:
        text = text[:max_chars]
    try:
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        print("⚠️ Embedding error:", e)
        return None

# Search in ChromaDB
def search_similar_posts(query: str, top_k=3):
    embedding = get_text_embedding(query)
    if not embedding:
        return {"answer": "Could not generate embedding.", "links": []}

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Format the answer and links
        answer = documents[0] if documents else "No relevant results found."
        links = []
        for meta in metadatas:
            if "topic" in meta and "created_at" in meta and "images" in meta:
                post_id = meta.get("post_id", "")
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{meta['topic'].replace(' ', '-')}/{post_id}"
                links.append({
                    "url": url,
                    "text": f"{meta['topic']} - {meta['created_at']}"
                })

        return {"answer": answer, "links": links}

    except Exception as e:
        print("⚠️ ChromaDB search failed:", e)
        return {"answer": "Search failed due to internal error.", "links": []}
