from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

app = FastAPI(title="MemoryShield")


# Load the trained MemoryShield model
MODEL_PATH = "./memoryshield_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


class MemoryRequest(BaseModel):
    memory: str


@app.get("/")
def home():
    return {"status": "MemoryShield is running"}


@app.post("/memory")
def check_memory(request: MemoryRequest):

    start_time = time.perf_counter()

    # Tokenization
    token_start = time.perf_counter()

    inputs = tokenizer(
        request.memory,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    tokenization_time = time.perf_counter() - token_start

    # Model inference
    inference_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    inference_time = time.perf_counter() - inference_start

    # Prediction
    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()

    if prediction == 1:
        status = "blocked"
    else:
        status = "allowed"

    total_time = time.perf_counter() - start_time

    return {
        "status": status,
        "prediction": "poisoned" if prediction == 1 else "clean",
        "confidence": round(confidence, 4),
        "tokenization_ms": round(tokenization_time * 1000, 2),
        "inference_ms": round(inference_time * 1000, 2),
        "total_processing_ms": round(total_time * 1000, 2),
        "memory": request.memory
    }