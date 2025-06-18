import os
import dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

# Load environment variables
dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"), base_url="https://aipipe.org/openai/v1")
clip_model = SentenceTransformer("clip-ViT-B-32")

def get_openai_text_embedding(text: str, max_chars: int = 7000):
    if len(text) > max_chars:
        text = text[:max_chars]
    try:
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        print("⚠️ Embedding error:", e)
        return None

def get_clip_image_embedding(image: Image.Image):
    try:
        embedding = clip_model.encode(image, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print("⚠️ Image embedding error:", e)
        return None
