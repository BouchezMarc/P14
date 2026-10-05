import json
import re
import sys
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

TEST_FILE = Path("data/processed/sft_test.jsonl")

OUTPUT_DIR = Path("results/baseline")
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.jsonl"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 1024

# Baseline déterministe
DO_SAMPLE = False

# Qwen3-1.7B sur RTX 5080
TORCH_DTYPE = torch.bfloat16


# ============================================================
# OUTILS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalisation simple pour les métriques lexicales."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def token_f1(reference: str, prediction: str) -> float:
    """
    F1 lexical basé sur les tokens.

    Ce n'est PAS une mesure clinique.
    Il mesure uniquement le recouvrement lexical
    entre la référence et la génération.
    """

    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)

    common = sum((ref_counter & pred_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(reference: str, prediction: str) -> float:
    return float(
        normalize_text(reference) == normalize_text(prediction)
    )


def load_dataset(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"[ERREUR] JSON invalide ligne "
                    f"{line_number}: {e}",
                    file=sys.stderr,
                )

                continue

            if "instruction" not in record or "response" not in record:

                print(
                    f"[ATTENTION] Champs manquants ligne "
                    f"{line_number}",
                    file=sys.stderr,
                )

                continue

            records.append(record)

    return records


# ============================================================
# PROMPT BASE MODEL
# ============================================================

def build_baseline_prompt(instruction: str) -> str:
    """
    Prompt destiné au modèle BASE.

    IMPORTANT :
    Qwen3-1.7B-Base n'est pas un modèle instruct/chat.
    On ne doit donc PAS utiliser apply_chat_template().

    On demande simplement au modèle de compléter une réponse.
    """

    return (
        "Question:\n"
        f"{instruction.strip()}\n\n"
        "Answer:\n"
    )


# ============================================================
# GENERATION
# ============================================================

def generate_response(model, tokenizer, instruction: str) -> str:

    prompt = build_baseline_prompt(instruction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE QWEN3-1.7B-BASE")
    print("=" * 70)

    print(f"Modèle       : {MODEL_NAME}")
    print(f"Dataset      : {TEST_FILE}")

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA         : "
            f"{torch.version.cuda}"
        )

    else:

        print("GPU          : CPU")

    print()

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : {TEST_FILE}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA n'est pas disponible."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    records = load_dataset(TEST_FILE)

    print(
        f"Exemples chargés : {len(records)}"
    )

    print()

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("Chargement du tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("Chargement de Qwen3-1.7B-Base...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
    )

    model.eval()

    print("Modèle chargé.")

    print(
        f"Dtype        : {model.dtype}"
    )

    print()

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    total_f1 = 0.0
    total_exact = 0.0

    by_language = {}

    predictions = []

    print("=" * 70)
    print("GENERATION")
    print("=" * 70)

    for index, record in enumerate(
        records,
        start=1,
    ):

        instruction = record["instruction"]
        reference = record["response"]

        try:

            prediction = generate_response(
                model,
                tokenizer,
                instruction,
            )

        except RuntimeError as e:

            print()
            print("[ERREUR GENERATION]")
            print(
                f"ID : {record.get('id')}"
            )
            print(e)
            print()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise

        f1 = token_f1(
            reference,
            prediction,
        )

        exact = exact_match(
            reference,
            prediction,
        )

        language = record.get(
            "language",
            "unknown",
        )

        total_f1 += f1
        total_exact += exact

        if language not in by_language:

            by_language[language] = {
                "count": 0,
                "f1_sum": 0.0,
                "exact_sum": 0.0,
            }

        by_language[language]["count"] += 1
        by_language[language]["f1_sum"] += f1
        by_language[language]["exact_sum"] += exact

        predictions.append(
            {
                "id": record.get("id"),
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
                "language": language,
                "task": record.get("task"),
                "source": record.get("source"),
                "metrics": {
                    "token_f1": f1,
                    "exact_match": exact,
                },
            }
        )

        # ----------------------------------------------------
        # AFFICHAGE DES 3 PREMIERS EXEMPLES
        # ----------------------------------------------------

        if index <= 3:

            print()
            print("=" * 70)
            print(
                f"EXEMPLE {index}/{len(records)}"
            )
            print("=" * 70)

            print("\nPROMPT BASE :")
            print(
                build_baseline_prompt(
                    instruction
                )
            )

            print("\nRÉFÉRENCE :")
            print(reference)

            print("\nRÉPONSE QWEN :")
            print(prediction)

            print(
                f"\nToken F1 : {f1:.4f}"
            )

            print(
                f"Exact Match : {exact:.4f}"
            )

        elif index % 10 == 0 or index == len(records):

            print(
                f"[{index}/{len(records)}] "
                f"F1={f1:.4f}"
            )

    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    count = len(records)

    metrics = {
        "model": MODEL_NAME,
        "dataset": str(TEST_FILE),
        "num_examples": count,
        "evaluation_type": "pre_sft_baseline",
        "prompt_format": (
            "Question:\\n{instruction}\\n\\nAnswer:\\n"
        ),
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "do_sample": DO_SAMPLE,
            "torch_dtype": str(TORCH_DTYPE),
        },
        "global": {
            "token_f1": (
                total_f1 / count
                if count
                else 0.0
            ),
            "exact_match": (
                total_exact / count
                if count
                else 0.0
            ),
        },
        "by_language": {},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
        },
    }

    for language, values in by_language.items():

        lang_count = values["count"]

        metrics["by_language"][language] = {

            "count": lang_count,

            "token_f1": (
                values["f1_sum"] / lang_count
                if lang_count
                else 0.0
            ),

            "exact_match": (
                values["exact_sum"] / lang_count
                if lang_count
                else 0.0
            ),
        }

    # --------------------------------------------------------
    # SAUVEGARDE PREDICTIONS
    # --------------------------------------------------------

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for prediction in predictions:

            f.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # SAUVEGARDE METRIQUES
    # --------------------------------------------------------

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("RESULTATS BASELINE")
    print("=" * 70)

    print(
        f"Exemples       : {count}"
    )

    print(
        f"Token F1       : "
        f"{metrics['global']['token_f1']:.4f}"
    )

    print(
        f"Exact Match    : "
        f"{metrics['global']['exact_match']:.4f}"
    )

    print()
    print("Par langue :")

    for language, values in metrics["by_language"].items():

        print(
            f"  {language}: "
            f"{values['count']} exemples | "
            f"F1={values['token_f1']:.4f} | "
            f"EM={values['exact_match']:.4f}"
        )

    print()
    print(
        f"Predictions : {PREDICTIONS_FILE}"
    )

    print(
        f"Metrics     : {METRICS_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()