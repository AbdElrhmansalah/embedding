from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()

def process(input_texts):
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    return embeddings
@app.post("/")
async def text(text: dict):
    string = text["text"]
    embeddings = process(string)
    return {"vector": embeddings}
