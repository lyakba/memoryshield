from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import requests

app = FastAPI(title="MemoryShield")

MODEL_PATH = "./memoryshield_model"
MEMORY_SERVICE_URL = "http://127.0.0.1:8001/store"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class MemoryRequest(BaseModel):
    memory: str


@app.get("/")
def home():
    return {
        "status": "MemoryShield is running",
        "memory_service": MEMORY_SERVICE_URL
    }


@app.post("/memory")
def check_memory(request: MemoryRequest):

    start_time = time.perf_counter()

    # -------------------------
    # 1. Tokenization
    # -------------------------

    token_start = time.perf_counter()

    inputs = tokenizer(
        request.memory,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    tokenization_time = time.perf_counter() - token_start

    # -------------------------
    # 2. AI inference
    # -------------------------

    inference_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    inference_time = time.perf_counter() - inference_start

    # -------------------------
    # 3. Classification
    # -------------------------

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item()

    # -------------------------
    # 4. Block poisoned memory
    # -------------------------

    if prediction == 1:

        total_time = time.perf_counter() - start_time

        return {
            "status": "blocked",
            "prediction": "poisoned",
            "confidence": round(confidence, 4),
            "tokenization_ms": round(tokenization_time * 1000, 2),
            "inference_ms": round(inference_time * 1000, 2),
            "total_processing_ms": round(total_time * 1000, 2),
            "memory": request.memory
        }

    # -------------------------
    # 5. Forward clean memory
    # -------------------------

    forward_start = time.perf_counter()

    try:

        response = requests.post(
            MEMORY_SERVICE_URL,
            json={
                "memory": request.memory
            },
            timeout=5
        )

        forward_time = time.perf_counter() - forward_start

        response.raise_for_status()

        memory_service_response = response.json()

        status = "allowed"

    except requests.RequestException as e:

        forward_time = time.perf_counter() - forward_start

        return {
            "status": "error",
            "prediction": "clean",
            "confidence": round(confidence, 4),
            "error": str(e)
        }

    # -------------------------
    # 6. Final response
    # -------------------------

    total_time = time.perf_counter() - start_time

    return {
        "status": status,
        "prediction": "clean",
        "confidence": round(confidence, 4),
        "tokenization_ms": round(tokenization_time * 1000, 2),
        "inference_ms": round(inference_time * 1000, 2),
        "forwarding_ms": round(forward_time * 1000, 2),
        "total_processing_ms": round(total_time * 1000, 2),
        "memory": request.memory,
        "memory_service": memory_service_response
    }