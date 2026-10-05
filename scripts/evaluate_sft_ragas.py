import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    FactualCorrectness,
    ContextRecall,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    default="predictions.jsonl",
)

parser.add_argument(
    "--output",
    default="results/sft_ragas.jsonl",
)

parser.add_argument(
    "--limit",
    type=int,
    default=100,
)

parser.add_argument(
    "--evaluator-model",
    default="qwen3-8b",
)

parser.add_argument(
    "--evaluator-url",
    default="http://localhost:1234/v1",
)

args = parser.parse_args()


# ============================================================
# LLM ÉVALUATEUR
# ============================================================

client = AsyncOpenAI(
    base_url=args.evaluator_url,
    api_key="lm-studio",
)

evaluator_llm = llm_factory(
    args.evaluator_model,
    client=client,
)


# ============================================================
# EMBEDDINGS MULTILINGUES
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device="cuda",
)


# ============================================================
# MÉTRIQUES
# ============================================================

async def prepare_metrics(language):

    faithfulness = Faithfulness(
        llm=evaluator_llm,
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    factual_correctness = FactualCorrectness(
        llm=evaluator_llm,
        mode="f1",
    )

    completeness = ContextRecall(
        llm=evaluator_llm,
    )

    # --------------------------------------------------------
    # Adaptation française
    # --------------------------------------------------------

    if language == "fr":

        faithfulness.prompt = await faithfulness.prompt.adapt(
            target_language="french",
            llm=evaluator_llm,
            adapt_instruction=True,
        )

        answer_relevancy.prompt = await answer_relevancy.prompt.adapt(
            target_language="french",
            llm=evaluator_llm,
            adapt_instruction=True,
        )

        factual_correctness.prompt = await factual_correctness.prompt.adapt(
            target_language="french",
            llm=evaluator_llm,
            adapt_instruction=True,
        )

        if hasattr(factual_correctness, "nli_prompt"):
            factual_correctness.nli_prompt = (
                await factual_correctness.nli_prompt.adapt(
                    target_language="french",
                    llm=evaluator_llm,
                    adapt_instruction=True,
                )
            )

        completeness.prompt = await completeness.prompt.adapt(
            target_language="french",
            llm=evaluator_llm,
            adapt_instruction=True,
        )

    return (
        faithfulness,
        answer_relevancy,
        factual_correctness,
        completeness,
    )


# ============================================================
# CHARGEMENT predictions.jsonl
# ============================================================

records = []

with open(args.input, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():
            records.append(json.loads(line))


records = records[:args.limit]

print("=" * 70)
print("ÉVALUATION SFT — RAGAS")
print("=" * 70)
print(f"Fichier   : {args.input}")
print(f"Exemples  : {len(records)}")
print()


# ============================================================
# CACHE DES MÉTRIQUES FR / EN
# ============================================================

metrics_cache = {}


# ============================================================
# ÉVALUATION
# ============================================================

async def evaluate_one(record):

    question = record["instruction"]
    reference = record["reference"]
    prediction = record["prediction"]

    language = record.get("language", "en").lower()

    if language not in ("fr", "en"):
        language = "en"

    # --------------------------------------------------------
    # Préparation métriques de la langue
    # --------------------------------------------------------

    if language not in metrics_cache:

        metrics_cache[language] = await prepare_metrics(
            language
        )

    (
        faithfulness,
        answer_relevancy,
        factual_correctness,
        completeness,
    ) = metrics_cache[language]

    # --------------------------------------------------------
    # FAITHFULNESS
    #
    # La référence constitue le contexte de vérité.
    # --------------------------------------------------------

    faith = await faithfulness.ascore(
        user_input=question,
        response=prediction,
        retrieved_contexts=[reference],
    )

    # --------------------------------------------------------
    # ANSWER RELEVANCY
    # --------------------------------------------------------

    relevancy = await answer_relevancy.ascore(
        user_input=question,
        response=prediction,
    )

    # --------------------------------------------------------
    # FACTUAL CORRECTNESS
    # --------------------------------------------------------

    correctness = await factual_correctness.ascore(
        response=prediction,
        reference=reference,
    )

    # --------------------------------------------------------
    # COMPLETENESS
    #
    # On mesure quelle partie de la référence est couverte
    # par la réponse du modèle.
    # --------------------------------------------------------

    complete = await completeness.ascore(
        user_input=question,
        retrieved_contexts=[prediction],
        reference=reference,
    )

    return {
        "id": record.get("id"),
        "language": language,
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "faithfulness": faith.value,
        "answer_relevancy": relevancy.value,
        "factual_correctness": correctness.value,
        "completeness": complete.value,
    }


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

async def main():

    results = []

    for i, record in enumerate(records, 1):

        print(
            f"\rÉvaluation : {i}/{len(records)}",
            end="",
            flush=True,
        )

        try:

            result = await evaluate_one(record)
            results.append(result)

        except Exception as exc:

            print()
            print(
                f"ERREUR {record.get('id', i)} : {exc}"
            )

    print()
    print()

    if not results:
        print("Aucun résultat.")
        return

    # ========================================================
    # SAUVEGARDE PAR EXEMPLE
    # ========================================================

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # MOYENNES
    # ========================================================

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
        "completeness",
    ]

    overall = {}

    for metric in metric_names:

        values = [
            r[metric]
            for r in results
            if r[metric] is not None
        ]

        overall[metric] = mean(values)


    # ========================================================
    # MOYENNES FR / EN
    # ========================================================

    by_language = {}

    for language in ("fr", "en"):

        subset = [
            r
            for r in results
            if r["language"] == language
        ]

        if not subset:
            continue

        by_language[language] = {}

        for metric in metric_names:

            by_language[language][metric] = mean(
                r[metric]
                for r in subset
            )


    # ========================================================
    # RÉSUMÉ
    # ========================================================

    summary = {
        "model": "sft",
        "input": args.input,
        "examples": len(results),
        "metrics": overall,
        "metrics_by_language": by_language,
    }

    summary_path = output_path.with_suffix(
        ".summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # AFFICHAGE
    # ========================================================

    print("=" * 70)
    print("RÉSULTATS SFT")
    print("=" * 70)

    print(
        f"Faithfulness        : {overall['faithfulness']:.4f}"
    )

    print(
        f"Answer Relevancy    : {overall['answer_relevancy']:.4f}"
    )

    print(
        f"Factual Correctness : {overall['factual_correctness']:.4f}"
    )

    print(
        f"Completeness        : {overall['completeness']:.4f}"
    )

    print()

    for language, metrics in by_language.items():

        print(f"--- {language.upper()} ---")

        for metric, value in metrics.items():

            print(
                f"{metric:22s}: {value:.4f}"
            )

        print()

    print(f"Détails : {output_path}")
    print(f"Résumé  : {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())