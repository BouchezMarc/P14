from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import uuid
import requests

VLLM_URL = os.getenv("VLLM_URL", "http://p14-vllm:8000")

app = FastAPI(
    title="P14 Medical AI API",
    version="0.1.0",
)


class TriageRequest(BaseModel):
    question: str
    max_tokens: int = 256
    temperature: float = 0.0


class TriageResponse(BaseModel):
    request_id: str
    model: str
    response: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": app.version,
        "model": "dpo_v3",
    }


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    prompt = (
        "<|user|>\n"
        f"{request.question}\n"
        "<|assistant|>\n"
    )

    payload = {
        "model": "dpo_v3",
        "prompt": prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM unavailable: {exc}",
        )

    data = response.json()
    latency_ms = (time.perf_counter() - start) * 1000

    return TriageResponse(
        request_id=request_id,
        model="dpo_v3",
        response=data["choices"][0]["text"],
        latency_ms=round(latency_ms, 2),
    )