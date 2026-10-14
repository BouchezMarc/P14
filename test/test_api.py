from unittest.mock import patch, Mock

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version():
    response = client.get("/version")

    assert response.status_code == 200

    data = response.json()

    assert data["api_version"] == "0.1.0"
    assert data["model"] == "dpo_v8"


@patch("src.api.requests.post")
def test_triage(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": "Test medical response."
            }
        ]
    }

    mock_post.return_value = mock_response

    response = client.post(
        "/api/v1/triage",
        json={
            "question": "What are the symptoms of iron deficiency anemia?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert data["model"] == "dpo_v8"
    assert data["response"] == "Test medical response."
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0


@patch("src.api.requests.post")
def test_triage_custom_parameters(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "text": "Test response."
            }
        ]
    }

    mock_post.return_value = mock_response

    response = client.post(
        "/api/v1/triage",
        json={
            "question": "What is anemia?",
            "max_tokens": 128,
            "temperature": 0.2,
        },
    )

    assert response.status_code == 200

    mock_post.assert_called_once()

    payload = mock_post.call_args.kwargs["json"]

    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.2


def test_triage_invalid_request():
    response = client.post(
        "/api/v1/triage",
        json={},
    )

    assert response.status_code == 422


@patch(
    "src.api.compute_log_probability",
    side_effect=[-1.192527, -1.134541],
)
def test_rewardbench_compare(mock_scoring):
    response = client.post(
        "/api/v1/compare",
        json={
            "prompt": "What should be done for severe chest pain?",
            "chosen": [
                {
                    "role": "user",
                    "content": "What should be done for severe chest pain?",
                },
                {
                    "role": "assistant",
                    "content": "Seek emergency medical care immediately.",
                },
            ],
            "rejected": [
                {
                    "role": "user",
                    "content": "What should be done for severe chest pain?",
                },
                {
                    "role": "assistant",
                    "content": "Ignore the pain and wait.",
                },
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert data["model"] == "dpo_v8"
    assert data["chosen_score"] == -1.192527
    assert data["rejected_score"] == -1.134541
    assert data["preferred"] == "rejected"
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0

    assert mock_scoring.call_count == 2


@patch(
    "src.api.compute_log_probability",
    side_effect=[-0.8, -1.2],
)
def test_rewardbench_prefers_chosen(mock_scoring):
    response = client.post(
        "/api/v1/compare",
        json={
            "prompt": "What is anemia?",
            "chosen": [
                {"role": "assistant", "content": "Anemia is a low red blood cell count."},
            ],
            "rejected": [
                {"role": "assistant", "content": "Anemia is a broken bone."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["preferred"] == "chosen"


def test_rewardbench_invalid_request():
    response = client.post(
        "/api/v1/compare",
        json={
            "prompt": "What is anemia?",
            "chosen": [],
        },
    )

    assert response.status_code == 422
