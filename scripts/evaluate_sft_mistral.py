"""
Évaluation des réponses SFT avec Ministral 14B comme LLM-as-a-Judge.

SOURCE
    results/sft_v5/predictions.jsonl

SORTIES
    results/sft_v5/evaluation/mistral_judge_predictions.jsonl
    results/sft_v5/evaluation/mistral_judge_summary.json

VARIABLE D'ENVIRONNEMENT
    MISTRAL_API_KEY

INSTALLATION
    uv add mistralai

EXEMPLE
    uv run python scripts/evaluate_sft_mistral.py --limit 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "sft_v5" / "predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sft_v5" / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "mistral_judge_predictions.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "mistral_judge_summary.json"

MISTRAL_MODEL = "ministral-14b-2512"

SYSTEM_PROMPT = '''You are a strict evaluator of medical question-answering systems.

You evaluate a model prediction against a medical question and a reference answer.

IMPORTANT RULES:

* Evaluate the PREDICTION. Do not rewrite or improve it.
* Do not reward verbosity.
* Evaluate whether the prediction answers the QUESTION that was asked.
* Use the REFERENCE as the primary reference for the expected answer.
* Do not require exact wording or sentence structure.
* Equivalent medical wording should receive the same evaluation.
* Do not penalize a medically correct statement merely because it is not explicitly stated in the reference.
* However, unsupported additional claims should be penalized when they are medically incorrect, contradictory to the reference, misleading, or unnecessarily speculative.
* Penalize contradictions with the reference.
* Penalize wrong diseases, diagnoses, genes, chromosomes, mechanisms, causes, or other medical facts.
* Penalize invented causes or unsupported diagnoses.
* Penalize clinically misleading or dangerous recommendations when present.
* Penalize topic drift.
* A prediction can be medically plausible but still receive a low score if it does not answer the question asked.
* A prediction can contain correct additional information and still receive a high score if that information is medically accurate and relevant.
* Do not penalize a response simply because it contains more information than the reference.
* Do not expect unrelated information when the reference is short.
* If the reference is incomplete or ambiguous, do not assume that information absent from the reference is necessarily false.
* Judge the prediction based on the question, the reference, and the medical correctness of the claims.

Evaluate four criteria.

1. FAITHFULNESS

Does the prediction remain faithful to the information supported by the reference and avoid introducing misleading claims?

Penalize:

* contradictions with the reference;
* invented or unsupported medical facts when they are misleading or speculative;
* wrong diseases, genes, chromosomes, mechanisms, or causes;
* unsupported diagnoses;
* topic drift;
* claims that materially distort the meaning of the reference.

Do not penalize medically correct additional information solely because it is absent from the reference.

2. ANSWER RELEVANCY

Does the prediction directly answer the question asked?

Penalize:

* answering a different question;
* discussing a related but different disease or condition;
* excessive unrelated information;
* failure to address the main question.

A concise answer can receive a high score if it directly answers the question.

3. CORRECTNESS

Are the medical statements in the prediction medically correct and consistent with the reference?

Penalize strongly:

* major medical errors;
* wrong diseases or diagnoses;
* wrong genes or chromosomes;
* wrong mechanisms or causes;
* contradictions;
* fabricated medical facts;
* clinically misleading or dangerous recommendations.

Do not mark a statement incorrect merely because it is not explicitly present in the reference. When a statement goes beyond the reference, assess whether it is medically accurate, relevant, and consistent with the question.

4. COMPLETENESS

Does the prediction contain the important information needed to adequately answer the question?

Consider the reference as the main indication of the expected content.

Penalize:

* omission of important information required to answer the question;
* incomplete explanations when key elements from the reference are missing;
* answers that mention the topic but fail to provide the requested information.

Do not require unnecessary details or verbosity.

SCORING SCALE

0 = completely wrong, absent, or unrelated
1 = severely deficient; major errors or failure to answer
2 = substantially deficient; important errors or omissions
3 = partially correct; answers the question but has notable errors or omissions
4 = mostly correct; minor errors or omissions
5 = correct, relevant, faithful, and sufficiently complete

IMPORTANT SCORING PRINCIPLE:

The four criteria are independent.

A prediction may be:

* medically correct but incomplete;
* complete but contain a factual error;
* relevant but medically incorrect;
* medically correct but irrelevant to the question.

Score each criterion independently.

Return ONLY valid JSON with exactly this structure:

{
"faithfulness": {"score": 0, "reason": "short reason"},
"answer_relevancy": {"score": 0, "reason": "short reason"},
"correctness": {"score": 0, "reason": "short reason"},
"completeness": {"score": 0, "reason": "short reason"}
}
'''


def build_user_prompt(question: str, reference: str, prediction: str) -> str:
    return f"""Evaluate this medical QA response.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

MODEL PREDICTION:
{prediction}

Evaluate the prediction using the four criteria defined in the system prompt.

Return only the required JSON object."""


def parse_json_response(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    data = json.loads(content)

    required = {
        "faithfulness",
        "answer_relevancy",
        "correctness",
        "completeness",
    }

    if set(data.keys()) != required:
        raise ValueError(f"Clés JSON inattendues : {sorted(data.keys())}")

    for criterion in required:
        value = data[criterion]

        if not isinstance(value, dict):
            raise ValueError(f"{criterion} doit être un objet")

        score = value.get("score")
        reason = value.get("reason")

        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 5:
            raise ValueError(f"Score invalide pour {criterion}: {score!r}")

        if not isinstance(reason, str):
            raise ValueError(f"Reason invalide pour {criterion}")

    return data


def evaluate_one(client: Mistral, item: dict) -> dict:
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    item["instruction"],
                    item["reference"],
                    item["prediction"],
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=500,
        random_seed=42,
        #reasoning_effort="medium",
    )

    evaluation = parse_json_response(response.choices[0].message.content)

    scores = [
        evaluation["faithfulness"]["score"],
        evaluation["answer_relevancy"]["score"],
        evaluation["correctness"]["score"],
        evaluation["completeness"]["score"],
    ]

    return {
        "faithfulness": evaluation["faithfulness"],
        "answer_relevancy": evaluation["answer_relevancy"],
        "correctness": evaluation["correctness"],
        "completeness": evaluation["completeness"],
        "overall": round(mean(scores), 3),
    }


def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_number}: {exc}"
                ) from exc

            for field in ["id", "instruction", "reference", "prediction", "language"]:
                if field not in item:
                    raise ValueError(
                        f"Champ '{field}' absent dans {path}, ligne {line_number}"
                    )

            records.append(item)

    return records


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"examples": 0}

    def avg(path: tuple[str, ...]) -> float:
        values = []
        for item in results:
            value = item
            for key in path:
                value = value[key]
            values.append(value)
        return round(mean(values), 3)

    summary = {
        "examples": len(results),
        "judge_model": MISTRAL_MODEL,
        "criteria": {
            "faithfulness": avg(("evaluation", "faithfulness", "score")),
            "answer_relevancy": avg(("evaluation", "answer_relevancy", "score")),
            "correctness": avg(("evaluation", "correctness", "score")),
            "completeness": avg(("evaluation", "completeness", "score")),
            "overall": avg(("evaluation", "overall")),
        },
    }

    by_language = defaultdict(list)
    for item in results:
        by_language[item.get("language", "unknown")].append(item)

    summary["by_language"] = {}

    for language, language_items in sorted(by_language.items()):
        summary["by_language"][language] = {
            "examples": len(language_items),
            "faithfulness": round(
                mean(x["evaluation"]["faithfulness"]["score"] for x in language_items), 3
            ),
            "answer_relevancy": round(
                mean(x["evaluation"]["answer_relevancy"]["score"] for x in language_items), 3
            ),
            "correctness": round(
                mean(x["evaluation"]["correctness"]["score"] for x in language_items), 3
            ),
            "completeness": round(
                mean(x["evaluation"]["completeness"]["score"] for x in language_items), 3
            ),
            "overall": round(
                mean(x["evaluation"]["overall"] for x in language_items), 3
            ),
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Évalue predictions.jsonl avec Ministral 14B."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERREUR : MISTRAL_API_KEY est absente de l'environnement.", file=sys.stderr)
        return 1

    try:
        items = load_predictions(args.source)
    except Exception as exc:
        print(f"ERREUR lecture source : {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        items = items[:args.limit]

    if not items:
        print("Aucun exemple à évaluer.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    client = Mistral(api_key=api_key)

    print("=" * 70)
    print("ÉVALUATION SFT — MISTRAL LLM-AS-A-JUDGE")
    print("=" * 70)
    print(f"Source      : {args.source}")
    print(f"Exemples    : {len(items)}")
    print(f"Modèle juge : {MISTRAL_MODEL}")
    print(f"Résultats   : {args.output}")
    print(f"Résumé      : {args.summary}")
    print("=" * 70)

    results = []
    errors = []

    with args.output.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(items, start=1):
            print(f"[{index:>4}/{len(items)}] {item['id']} — {item['language']}")

            try:
                evaluation = evaluate_one(client, item)

                result = {
                    "id": item["id"],
                    "language": item["language"],
                    "task": item.get("task"),
                    "instruction": item["instruction"],
                    "reference": item["reference"],
                    "prediction": item["prediction"],
                    "evaluation": evaluation,
                }

                results.append(result)

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                print(
                    f"      F={evaluation['faithfulness']['score']} "
                    f"R={evaluation['answer_relevancy']['score']} "
                    f"C={evaluation['correctness']['score']} "
                    f"Co={evaluation['completeness']['score']} "
                    f"→ {evaluation['overall']:.2f}/5"
                )

            except Exception as exc:
                error = {
                    "id": item["id"],
                    "error": str(exc),
                }
                errors.append(error)

                print(f"      ERREUR : {exc}", file=sys.stderr)
                print(f"      BODY    : {getattr(exc, 'body', None)}", file=sys.stderr)
                print(f"      HEADERS : {getattr(exc, 'headers', None)}", file=sys.stderr)

            if args.delay > 0 and index < len(items):
                time.sleep(args.delay)

    summary = summarize(results)
    summary["source_file"] = str(args.source)
    summary["output_file"] = str(args.output)
    summary["errors"] = errors

    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Évalués : {summary['examples']}")
    print(f"Erreurs : {len(errors)}")

    if results:
        criteria = summary["criteria"]
        print(f"Faithfulness     : {criteria['faithfulness']:.3f}/5")
        print(f"Answer Relevancy : {criteria['answer_relevancy']:.3f}/5")
        print(f"Correctness      : {criteria['correctness']:.3f}/5")
        print(f"Completeness     : {criteria['completeness']:.3f}/5")
        print(f"GLOBAL           : {criteria['overall']:.3f}/5")

    print()
    print(f"Résultats : {args.output}")
    print(f"Résumé    : {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())