from sentence_transformers import SentenceTransformer
from fastapi import FastAPI


app = FastAPI()
model = SentenceTransformer('intfloat/multilingual-e5-large')
def process(input_texts):
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    return embeddings
@app.post("/")
async def text(text: dict):
    string = text["text"]
    embeddings = process(string)
    return {"vector": embeddings}
