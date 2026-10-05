import json
import os
import re
import string

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sft",
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sft_test.jsonl",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "sft",
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions.jsonl",
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "metrics.json",
)

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512


# ================================================================
# UTILITAIRES
# ================================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def normalize_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_f1(prediction, reference):
    pred_tokens = normalize_text(
        prediction
    ).split()

    ref_tokens = normalize_text(
        reference
    ).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}

    for token in pred_tokens:
        pred_counts[token] = (
            pred_counts.get(token, 0) + 1
        )

    ref_counts = {}

    for token in ref_tokens:
        ref_counts[token] = (
            ref_counts.get(token, 0) + 1
        )

    common = 0

    for token in pred_counts:
        if token in ref_counts:
            common += min(
                pred_counts[token],
                ref_counts[token],
            )

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def exact_match(prediction, reference):
    return int(
        normalize_text(prediction)
        == normalize_text(reference)
    )


def build_prompt(instruction):
    return (
        "Question:\n"
        f"{instruction}\n\n"
        "Answer:\n"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("ÉVALUATION SFT — MODÈLE LoRA SAUVEGARDÉ")
    print("=" * 70)

    print(f"Base model       : {BASE_MODEL}")
    print(f"Modèle LoRA      : {MODEL_PATH}")
    print(f"Test             : {TEST_FILE}")
    print(f"Sortie           : {OUTPUT_DIR}")
    print()

    # ============================================================
    # VERIFICATION DES FICHIERS
    # ============================================================

    print("=" * 70)
    print("VÉRIFICATION DES FICHIERS")
    print("=" * 70)

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle LoRA introuvable :\n{MODEL_PATH}"
        )

    adapter_config = os.path.join(
        MODEL_PATH,
        "adapter_config.json",
    )

    adapter_model = os.path.join(
        MODEL_PATH,
        "adapter_model.safetensors",
    )

    tokenizer_file = os.path.join(
        MODEL_PATH,
        "tokenizer.json",
    )

    tokenizer_config = os.path.join(
        MODEL_PATH,
        "tokenizer_config.json",
    )

    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(
            "Fichier adapter_config.json introuvable :\n"
            f"{adapter_config}"
        )

    if not os.path.isfile(adapter_model):
        raise FileNotFoundError(
            "Fichier adapter_model.safetensors introuvable :\n"
            f"{adapter_model}"
        )

    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(
            "Fichier tokenizer.json introuvable :\n"
            f"{tokenizer_file}"
        )

    if not os.path.isfile(tokenizer_config):
        raise FileNotFoundError(
            "Fichier tokenizer_config.json introuvable :\n"
            f"{tokenizer_config}"
        )

    if not os.path.isfile(TEST_FILE):
        raise FileNotFoundError(
            f"Fichier de test introuvable :\n{TEST_FILE}"
        )

    print("Modèle LoRA      : OK")
    print("adapter_config   : OK")
    print("adapter_model    : OK")
    print("tokenizer.json   : OK")
    print("tokenizer_config : OK")
    print("Dataset test     : OK")
    print()

    # ============================================================
    # GPU
    # ============================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible."
        )

    print("=" * 70)
    print("ENVIRONNEMENT")
    print("=" * 70)

    print(
        f"GPU              : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA             : "
        f"{torch.version.cuda}"
    )

    print(
        f"PyTorch          : "
        f"{torch.__version__}"
    )

    print()

    # ============================================================
    # TEST DATASET
    # ============================================================

    print("=" * 70)
    print("CHARGEMENT DU TEST")
    print("=" * 70)

    records = load_jsonl(
        TEST_FILE
    )

    print(
        f"Nombre de tests  : "
        f"{len(records)}"
    )

    if len(records) != 500:
        raise RuntimeError(
            "Le test doit contenir exactement "
            f"500 exemples. Trouvé : {len(records)}"
        )

    print()

    # ============================================================
    # CHARGEMENT DU TOKENIZER SAUVEGARDÉ
    # ============================================================

    print("=" * 70)
    print("CHARGEMENT DU TOKENIZER SAUVEGARDÉ")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print(
        f"Tokenizer chargé : {MODEL_PATH}"
    )

    # ============================================================
    # CHARGEMENT DU MODÈLE
    # ============================================================

    print("=" * 70)
    print("CHARGEMENT DU MODÈLE")
    print("=" * 70)

    model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print("Base model chargé.")

    # ------------------------------------------------------------
    # Chargement du LoRA sauvegardé
    # ------------------------------------------------------------

    model.load_adapter(
        MODEL_PATH
    )

    print(
        "LoRA chargé       : "
        f"{MODEL_PATH}"
    )

    # ------------------------------------------------------------
    # Mode inférence Unsloth
    # ------------------------------------------------------------

    FastLanguageModel.for_inference(
        model
    )

    model.eval()

    print("Mode inférence    : activé")
    print()

    # ============================================================
    # GENERATION
    # ============================================================

    print("=" * 70)
    print("GÉNÉRATION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    predictions = []

    total_f1 = 0.0
    total_em = 0

    for index, example in enumerate(
        records,
        start=1,
    ):

        prompt = build_prompt(
            example["instruction"]
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                model.device
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            outputs[0][input_length:]
        )

        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        reference = example["response"]

        f1 = token_f1(
            prediction,
            reference,
        )

        em = exact_match(
            prediction,
            reference,
        )

        total_f1 += f1
        total_em += em

        result = {
            "id": example["id"],
            "instruction": example[
                "instruction"
            ],
            "reference": reference,
            "prediction": prediction,
            "language": example.get(
                "language"
            ),
            "task": example.get(
                "task"
            ),
            "source": example.get(
                "source"
            ),
            "token_f1": f1,
            "exact_match": em,
        }

        predictions.append(
            result
        )

        if (
            index == 1
            or index % 25 == 0
        ):
            print(
                f"{index:03d}/500 | "
                f"F1 moyen="
                f"{total_f1 / index:.4f} | "
                f"EM moyen="
                f"{total_em / index:.4f}"
            )

    # ============================================================
    # METRIQUES FINALES
    # ============================================================

    mean_f1 = (
        total_f1 / len(records)
    )

    mean_em = (
        total_em / len(records)
    )

    metrics = {
        "model": BASE_MODEL,
        "adapter": "models/sft",

        "test": {
            "file": (
                "data/processed/"
                "sft_test.jsonl"
            ),
            "examples": len(records),
        },

        "generation": {
            "max_seq_length": (
                MAX_SEQ_LENGTH
            ),
            "max_new_tokens": (
                MAX_NEW_TOKENS
            ),
            "do_sample": False,
        },

        "metrics": {
            "token_f1": mean_f1,
            "exact_match": mean_em,
        },
    }

    # ============================================================
    # SAUVEGARDE DES PREDICTIONS
    # ============================================================

    with open(
        PREDICTIONS_FILE,
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

    # ============================================================
    # SAUVEGARDE DES METRIQUES
    # ============================================================

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ============================================================
    # RESULTATS
    # ============================================================

    print()
    print("=" * 70)
    print("RÉSULTATS SFT")
    print("=" * 70)

    print(
        f"Exemples       : "
        f"{len(records)}"
    )

    print(
        f"Token F1       : "
        f"{mean_f1:.4f}"
    )

    print(
        f"Exact Match    : "
        f"{mean_em:.4f}"
    )

    print()

    print(
        f"Predictions    : "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Metrics        : "
        f"{METRICS_FILE}"
    )

    print()
    print("=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
