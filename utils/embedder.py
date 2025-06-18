import os
import requests
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"), base_url="https://aipipe.org/openai/v1")

# Hugging Face API token (store this in your env)
HF_TOKEN = os.getenv("HF_TOKEN")

def get_openai_text_embedding(text: str, max_chars: int = 7000):
    if len(text) > max_chars:
        text = text[:max_chars]
    try:
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        print("⚠️ Text embedding error:", e)
        return None

def get_clip_image_embedding(image: Image.Image):
    if HF_TOKEN is None:
        print("⚠️ HF_TOKEN not set")
        return None

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Use huggingface clip image model endpoint (or general inference endpoint)
    url = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
    payload = {
        "inputs": {"image": b64_image}
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list):
            return result  # already a vector
        else:
            print("⚠️ Unexpected HF response format")
            return None
    except requests.exceptions.RequestException as e:
        print("❌ HF API image embedding failed:", e)
        return None
