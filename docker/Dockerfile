FROM vllm/vllm-openai:latest

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn requests

COPY src/api.py /app/api.py
COPY docker/start.sh /app/start.sh
COPY models/dpo_v8 /models/dpo_v8

RUN chmod +x /app/start.sh

ENV VLLM_URL=http://127.0.0.1:8000
ENV MODEL_NAME=dpo_v8
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8080

ENTRYPOINT ["/app/start.sh"]