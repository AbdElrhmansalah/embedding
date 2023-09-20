from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

model = SentenceTransformer('intfloat/multilingual-e5-large')

app = FastAPI()

def process(input_texts):
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    return embeddings
@app.post("/embedText")
async def text(text: dict):
    string = text["text"]
    embeddings = process(string)
    return {"vector": embeddings}