from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load the model
model = SentenceTransformer('e5_local_model')

# Create a FastAPI app
app = FastAPI()

# Define a request model
class TextRequest(BaseModel):
    text: List[str]

# Process function to encode text
def process(input_texts: List[str]):
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    return embeddings

@app.post("/")
async def get_embeddings(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text field is required")
    embeddings = process(request.text)
    return {"embeddings": embeddings.tolist()}  # Convert numpy array to list for JSON serialization
