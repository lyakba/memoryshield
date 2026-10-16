from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Memory Service")

memory_storage = []


class MemoryRequest(BaseModel):
    memory: str


@app.get("/")
def home():
    return {
        "status": "Memory Service is running",
        "stored_memories": len(memory_storage)
    }


@app.post("/store")
def store_memory(request: MemoryRequest):
    entry = {
        "memory": request.memory,
        "stored_at": datetime.now().isoformat()
    }

    memory_storage.append(entry)

    return {
        "status": "stored",
        "memory": request.memory,
        "total_memories": len(memory_storage)
    }


@app.get("/memories")
def get_memories():
    return {
        "count": len(memory_storage),
        "memories": memory_storage
    }