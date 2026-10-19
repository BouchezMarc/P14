
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
import json
import os
import time
import uuid
import requests

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "dpo_v8")

# Répertoire de journalisation configurable par variable d'environnement.
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "p14_api.jsonl"
LOG_LOCK = Lock()


def write_log(event: dict) -> None:
    """Ajoute un événement JSON sur une nouvelle ligne du fichier local."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_LOCK:
            with LOG_FILE.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
    except OSError as exc:
        # Une panne de journalisation ne doit pas masquer la réponse API.
        print(f"[LOGGING ERROR] Impossible d'écrire {LOG_FILE}: {exc}")


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


class RewardBenchMessage(BaseModel):
    role: str
    content: str


class CompareRequest(BaseModel):
    prompt: str
    chosen: list[RewardBenchMessage]
    rejected: list[RewardBenchMessage]


class CompareResponse(BaseModel):
    request_id: str
    model: str
    chosen_score: float
    rejected_score: float
    preferred: Literal["chosen", "rejected", "tie"]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": app.version,
        "model": MODEL_NAME,
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
        "model": MODEL_NAME,
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
        data = response.json()

        answer = data["choices"][0]["text"]
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "triage",
            "status": "success",
            "request_id": request_id,
            "model": MODEL_NAME,
            "question": request.question,
            "response": answer,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "latency_ms": latency_ms,
        })

        return TriageResponse(
            request_id=request_id,
            model=MODEL_NAME,
            response=answer,
            latency_ms=latency_ms,
        )

    except requests.RequestException as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "triage",
            "status": "error",
            "request_id": request_id,
            "model": MODEL_NAME,
            "question": request.question,
            "latency_ms": latency_ms,
            "error": str(exc),
        })

        raise HTTPException(
            status_code=502,
            detail={
                "error": f"vLLM unavailable: {exc}",
                "latency_ms": latency_ms,
            },
        )

    except (ValueError, KeyError) as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "triage",
            "status": "error",
            "request_id": request_id,
            "model": MODEL_NAME,
            "question": request.question,
            "latency_ms": latency_ms,
            "error": str(exc),
        })

        raise HTTPException(
            status_code=502,
            detail="Invalid response returned by vLLM",
        )


MAX_SEQ_LENGTH = 2048


def extract_answer(messages: list[RewardBenchMessage]) -> str:
    for message in messages:
        if message.role == "assistant":
            return message.content

    raise HTTPException(
        status_code=422,
        detail="No assistant message found in candidate",
    )


def tokenize_text(text: str) -> list[int]:
    response = requests.post(
        f"{VLLM_URL}/tokenize",
        json={
            "model": MODEL_NAME,
            "prompt": text,
            "truncate_prompt_tokens": MAX_SEQ_LENGTH,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["tokens"]


def compute_log_probability(prompt: str, answer: str) -> float:
    prefix_tokens = tokenize_text(prompt + "\n")
    full_tokens = tokenize_text(prompt + "\n" + answer)

    if len(full_tokens) <= len(prefix_tokens):
        raise HTTPException(
            status_code=422,
            detail="No answer tokens available for scoring",
        )

    response = requests.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": full_tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]

    # Exclut les tokens du préfixe et le token généré en supplément.
    answer_logprobs = token_logprobs[
        len(prefix_tokens):len(full_tokens)
    ]

    scores = [score for score in answer_logprobs if score is not None]

    if not scores:
        raise HTTPException(
            status_code=502,
            detail="vLLM returned no answer token log-probabilities",
        )

    return sum(scores) / len(scores)


@app.post("/api/v1/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        chosen = extract_answer(request.chosen)
        rejected = extract_answer(request.rejected)

        chosen_score = compute_log_probability(request.prompt, chosen)
        rejected_score = compute_log_probability(request.prompt, rejected)

        if abs(chosen_score - rejected_score) < 1e-8:
            preferred = "tie"
        elif chosen_score > rejected_score:
            preferred = "chosen"
        else:
            preferred = "rejected"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "compare",
            "status": "success",
            "request_id": request_id,
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": round(chosen_score, 6),
            "rejected_score": round(rejected_score, 6),
            "preferred": preferred,
            "latency_ms": latency_ms,
        })

        return CompareResponse(
            request_id=request_id,
            model=MODEL_NAME,
            chosen_score=round(chosen_score, 6),
            rejected_score=round(rejected_score, 6),
            preferred=preferred,
            latency_ms=latency_ms,
        )

    except HTTPException as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "compare",
            "status": "error",
            "request_id": request_id,
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "latency_ms": latency_ms,
            "http_status": exc.status_code,
            "error": exc.detail,
        })
        raise

    except requests.RequestException as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "compare",
            "status": "error",
            "request_id": request_id,
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "latency_ms": latency_ms,
            "http_status": 502,
            "error": str(exc),
        })

        raise HTTPException(
            status_code=502,
            detail=f"vLLM scoring unavailable: {exc}",
        )

    except (ValueError, KeyError, TypeError) as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        write_log({
            "event": "compare",
            "status": "error",
            "request_id": request_id,
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "latency_ms": latency_ms,
            "http_status": 502,
            "error": str(exc),
        })

        raise HTTPException(
            status_code=502,
            detail="Invalid scoring response returned by vLLM",
        )
