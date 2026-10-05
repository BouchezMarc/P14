import argparse
import json
import os
import re
import string
import time

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
TEST_FILE = os.path.join(BASE_DIR, "data", "processed", "sft_test.jsonl")
MAX_SEQ_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 4
PRINT_EVERY = 25
MODEL_PATHS = {
    "base": None,
    "sft": os.path.join(BASE_DIR, "models", "sft_v5", "checkpoint-500"),
    "dpo": os.path.join(BASE_DIR, "models", "dpo_v3", "checkpoint-625"),
}
RESULTS_PATHS = {
    "base": os.path.join(BASE_DIR, "results", "base"),
    "sft": os.path.join(BASE_DIR, "results", "sft_v5"),
    "dpo": os.path.join(BASE_DIR, "results", "dpo_v3"),
}

SFT_SYSTEM_PROMPT = """You are a medical question-answering assistant.

Respond in the same language as the question.
Answer only the question asked.
Provide a clear, factual and clinically relevant answer.
Do not introduce unrelated diseases, conditions, or topics.
Do not invent medical facts, diagnoses, treatments, or recommendations.
When the available information is insufficient, state the uncertainty clearly.
Do not provide dangerous or unsupported medical recommendations."""

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
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = sum(
        min(pred_counts[token], ref_counts[token])
        for token in pred_counts
        if token in ref_counts
    )
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return int(normalize_text(prediction) == normalize_text(reference))


def build_prompt(instruction):
    return (
        f'{SFT_SYSTEM_PROMPT}\n\n'
        'Question:\n'
        f'{instruction}\n\n'
        'Answer:\n'
    )

def validate_model_path(model_name, model_path):
    if model_name == "base":
        return
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Modèle/adaptateur introuvable :\n{model_path}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = os.path.join(model_path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fichier requis introuvable :\n{path}")


def tokenizer_path_for(model_name, model_path):
    if model_name == "base":
        return BASE_MODEL
    if all(
        os.path.isfile(os.path.join(model_path, filename))
        for filename in ("tokenizer.json", "tokenizer_config.json")
    ):
        return model_path
    return BASE_MODEL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évaluation générique Base / SFT / DPO avec Token F1 et Exact Match."
    )
    parser.add_argument("--model", choices=("base", "sft", "dpo"), required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Limite temporaire pour un test rapide.")
    parser.add_argument("--force", action="store_true", help="Régénère même si predictions.jsonl existe.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size doit être >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens doit être >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit doit être >= 1")

    model_name = args.model
    model_path = MODEL_PATHS[model_name]
    # output_dir = os.path.join(BASE_DIR, "results", model_name)
    output_dir = RESULTS_PATHS[model_name]
    predictions_file = os.path.join(output_dir, "predictions.jsonl")
    metrics_file = os.path.join(output_dir, "metrics.json")

    print("=" * 70)
    print("ÉVALUATION GÉNÉRIQUE DU MODÈLE")
    print("=" * 70)
    print(f"Modèle évalué   : {model_name}")
    print(f"Base model      : {BASE_MODEL}")
    print(f"Adaptateur      : {model_path if model_path else 'aucun (Base)'}")
    print(f"Test            : {TEST_FILE}")
    print(f"Sortie          : {output_dir}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    if args.limit is not None:
        print(f"Limite test     : {args.limit}")
    print()

    if not os.path.isfile(TEST_FILE):
        raise FileNotFoundError(f"Fichier de test introuvable :\n{TEST_FILE}")
    validate_model_path(model_name, model_path)

    records = load_jsonl(TEST_FILE)
    if len(records) != 500:
        raise RuntimeError(
            f"Le test officiel doit contenir exactement 500 exemples. Trouvé : {len(records)}"
        )
    if args.limit is not None:
        records = records[:args.limit]

    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Nombre de tests  : {len(records)}")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible.")

    print("=" * 70)
    print("ENVIRONNEMENT")
    print("=" * 70)
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"CUDA             : {torch.version.cuda}")
    print(f"PyTorch          : {torch.__version__}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(predictions_file) and not args.force and args.limit is None:
        print("=" * 70)
        print("PRÉDICTIONS EXISTANTES")
        print("=" * 70)
        print(f"{predictions_file}")
        print("Utilisation du fichier existant. Utilisez --force pour régénérer.")
        return

    tokenizer_path = tokenizer_path_for(model_name, model_path)
    print("=" * 70)
    print("TOKENIZER")
    print("=" * 70)
    print(f"Chargement depuis : {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Le tokenizer n'a ni pad_token_id ni eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer chargé : OK")
    print()

    print("=" * 70)
    print("CHARGEMENT DU MODÈLE")
    print("=" * 70)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    print("Base model chargé : OK")

    if model_name != "base":
        model.load_adapter(model_path)
        print(f"LoRA chargé       : {model_path}")

    FastLanguageModel.for_inference(model)
    model.eval()

    # Ne conserve pas le max_length=32768 hérité de la configuration du modèle.
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = args.max_new_tokens
        model.generation_config.do_sample = False
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("Mode inférence    : activé")
    print()

    print("=" * 70)
    print("GÉNÉRATION")
    print("=" * 70)

    predictions = []
    total_f1 = 0.0
    total_em = 0
    total_generated_tokens = 0
    generation_start = time.perf_counter()

    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start:batch_start + args.batch_size]
        prompts = [build_prompt(example["instruction"]) for example in batch_records]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        padded_input_length = inputs["input_ids"].shape[1]
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )

            eos_id = tokenizer.eos_token_id

            print("\n" + "=" * 70)
            print("CONTROLE EOS")
            print("=" * 70)
            print("EOS token    :", repr(tokenizer.eos_token))
            print("EOS ID       :", eos_id)
            print("PAD token    :", repr(tokenizer.pad_token))
            print("PAD ID       :", tokenizer.pad_token_id)

            for i, output in enumerate(outputs[:5]):
                generated = output[inputs["input_ids"].shape[1]:]

                print(f"\nExemple {i + 1}")
                print("Derniers IDs :", generated[-10:].tolist())
                print("EOS présent  :", eos_id in generated.tolist())

                if eos_id in generated.tolist():
                    print("Position EOS  :", generated.tolist().index(eos_id))


        for batch_index, example in enumerate(batch_records):
            generated_tokens = outputs[batch_index, padded_input_length:]
            prediction = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            reference = example["response"]
            f1 = token_f1(prediction, reference)
            em = exact_match(prediction, reference)
            generated_token_count = int(generated_tokens.shape[0])

            total_f1 += f1
            total_em += em
            total_generated_tokens += generated_token_count

            predictions.append({
                "id": example["id"],
                "instruction": example["instruction"],
                "reference": reference,
                "prediction": prediction,
                "language": example.get("language"),
                "task": example.get("task"),
                "source": example.get("source"),
                "token_f1": f1,
                "exact_match": em,
                "generated_tokens": generated_token_count,
            })

        processed = min(batch_start + args.batch_size, len(records))
        if processed == len(records) or processed % PRINT_EVERY == 0:
            elapsed = time.perf_counter() - generation_start
            print(
                f"{processed:03d}/{len(records)} | "
                f"F1 moyen={total_f1 / processed:.4f} | "
                f"EM moyen={total_em / processed:.4f} | "
                f"tokens moy={total_generated_tokens / processed:.1f} | "
                f"{processed / elapsed:.2f} ex/s"
            )

    generation_time = time.perf_counter() - generation_start
    mean_f1 = total_f1 / len(records)
    mean_em = total_em / len(records)
    mean_generated_tokens = total_generated_tokens / len(records)

    metrics = {
        "model_type": model_name,
        "base_model": BASE_MODEL,
        "adapter": os.path.relpath(model_path, BASE_DIR) if model_path else None,
        "test": {
            "file": os.path.relpath(TEST_FILE, BASE_DIR),
            "examples": len(records),
        },
        "generation": {
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "do_sample": False,
        },
        "performance": {
            "generation_time_seconds": generation_time,
            "generation_time_minutes": generation_time / 60.0,
            "examples_per_second": len(records) / generation_time if generation_time > 0 else 0.0,
            "mean_generated_tokens": mean_generated_tokens,
            "total_generated_tokens": total_generated_tokens,
        },
        "metrics": {
            "token_f1": mean_f1,
            "exact_match": mean_em,
        },
    }

    with open(predictions_file, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"RÉSULTATS {model_name.upper()}")
    print("=" * 70)
    print(f"Exemples             : {len(records)}")
    print(f"Token F1             : {mean_f1:.4f}")
    print(f"Exact Match          : {mean_em:.4f}")
    print(f"Tokens générés moy.  : {mean_generated_tokens:.1f}")
    print(f"Tokens générés total : {total_generated_tokens}")
    print(f"Temps génération     : {generation_time / 60.0:.2f} min")
    print(f"Vitesse              : {len(records) / generation_time:.2f} ex/s")
    print()
    print(f"Predictions          : {predictions_file}")
    print(f"Metrics              : {metrics_file}")
    print()
    print("=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
