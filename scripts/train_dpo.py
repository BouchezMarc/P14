from __future__ import annotations

import argparse
import json
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel

import trl


# ---------------------------------------------------------------------------
# TRL 0.24.0 : MergeKit inutile pour ce DPO
# ---------------------------------------------------------------------------

mergekit_stub = types.ModuleType("trl.mergekit_utils")
mergekit_stub.__spec__ = importlib.machinery.ModuleSpec(
    "trl.mergekit_utils",
    loader=None,
)


class MergeConfig:
    pass


def merge_models(*args, **kwargs):
    raise RuntimeError("MergeKit non utilisé par ce script DPO.")


def upload_model_to_hf(*args, **kwargs):
    raise RuntimeError("MergeKit non utilisé par ce script DPO.")


mergekit_stub.MergeConfig = MergeConfig
mergekit_stub.merge_models = merge_models
mergekit_stub.upload_model_to_hf = upload_model_to_hf

sys.modules["trl.mergekit_utils"] = mergekit_stub


# ---------------------------------------------------------------------------
# TRL 0.24.0 : callbacks optionnels inutiles pour notre DPO
# ---------------------------------------------------------------------------

callbacks_stub = types.ModuleType("trl.trainer.callbacks")
callbacks_stub.__spec__ = importlib.machinery.ModuleSpec(
    "trl.trainer.callbacks",
    loader=None,
)


class SyncRefModelCallback:
    pass


callbacks_stub.SyncRefModelCallback = SyncRefModelCallback

sys.modules["trl.trainer.callbacks"] = callbacks_stub


from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel


BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
SFT_CHECKPOINT = Path("models/sft_v3/checkpoint-1000")

TRAIN_FILE = Path("data/processed/dataset_dpo_v2/anonymized/dpo_train.jsonl")
VALIDATION_FILE = Path("data/processed/dataset_dpo_v2/anonymized/dpo_validation.jsonl")

OUTPUT_DIR = Path("models/dpo_v4")

MAX_LENGTH = 2048


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_no}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def build_dataset(path: Path) -> Dataset:
    rows = load_jsonl(path)

    required_fields = {"prompt", "chosen", "rejected"}

    cleaned = []

    for index, row in enumerate(rows):

        missing = required_fields - row.keys()

        if missing:
            raise ValueError(
                f"{path}: exemple {index}: "
                f"champs manquants: {sorted(missing)}"
            )

        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        if not isinstance(prompt, str):
            raise ValueError(
                f"{path}: exemple {index}: `prompt` doit être une chaîne."
            )

        if not isinstance(chosen, list):
            raise ValueError(
                f"{path}: exemple {index}: `chosen` doit être une liste de messages."
            )

        if not isinstance(rejected, list):
            raise ValueError(
                f"{path}: exemple {index}: `rejected` doit être une liste de messages."
            )

        if not prompt.strip():
            raise ValueError(
                f"{path}: exemple {index}: `prompt` est vide."
            )

        if not chosen:
            raise ValueError(
                f"{path}: exemple {index}: `chosen` est vide."
            )

        if not rejected:
            raise ValueError(
                f"{path}: exemple {index}: `rejected` est vide."
            )

        for name, messages in (
            ("chosen", chosen),
            ("rejected", rejected),
        ):
            for message_index, message in enumerate(messages):

                if not isinstance(message, dict):
                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] doit être un objet."
                    )

                if "role" not in message or "content" not in message:
                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] doit contenir "
                        "`role` et `content`."
                    )

                if message["role"] not in {"user", "assistant", "system"}:
                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] rôle invalide: "
                        f"{message['role']!r}"
                    )

                if not isinstance(message["content"], str):
                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}]['content'] "
                        "doit être une chaîne."
                    )

        cleaned.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    return Dataset.from_list(cleaned)


def load_sft_model():

    if not SFT_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint SFT introuvable:\n"
            f"  {SFT_CHECKPOINT}"
        )

    print("=" * 70)
    print("LOADING SFT MODEL")
    print("=" * 70)

    print(f"Base model     : {BASE_MODEL}")
    print(f"SFT checkpoint : {SFT_CHECKPOINT}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing=True,
    )

    model = PeftModel.from_pretrained(
        model,
        str(SFT_CHECKPOINT),
        is_trainable=True,        
    )

    model.load_adapter(
        str(SFT_CHECKPOINT),
        adapter_name="reference",
    )

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|>\\n' }}"
            "{{ message['content'] + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\\n' }}"
            "{% endif %}"
    )

    return model, tokenizer


def main():

    parser = argparse.ArgumentParser(
        description="DPO training from the SFT v5 LoRA checkpoint."
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    print(f"PyTorch        : {torch.__version__}")
    print(f"TRL            : {trl.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM           : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 70)
    print("LOADING DPO DATASETS")
    print("=" * 70)

    train_dataset = build_dataset(TRAIN_FILE)
    validation_dataset = build_dataset(VALIDATION_FILE)

    print(f"DPO train      : {len(train_dataset)}")
    print(f"DPO validation : {len(validation_dataset)}")

    model, tokenizer = load_sft_model()

    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=args.learning_rate,

        weight_decay=0.01,
        warmup_ratio=0.05,

        lr_scheduler_type="cosine",

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        beta=args.beta,
        loss_type="sigmoid",

        max_length=MAX_LENGTH,
        truncation_mode="keep_end",

        logging_steps=10,

        report_to="mlflow",

        seed=args.seed,
        data_seed=args.seed,

        optim="adamw_torch_fused",

        dataloader_num_workers=0,

        remove_unused_columns=False,

        model_adapter_name="default",
        ref_adapter_name="reference",
    )

    print("=" * 70)
    print("INITIALIZING DPO TRAINER")
    print("=" * 70)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    print("=" * 70)
    print("STARTING DPO TRAINING")
    print("=" * 70)

    print(f"Epochs        : {args.epochs}")
    print(f"Learning rate : {args.learning_rate}")
    print(f"Beta          : {args.beta}")
    print(f"Train         : {len(train_dataset)}")
    print(f"Validation    : {len(validation_dataset)}")
    print("Loss          : sigmoid DPO")
    print("Reference     : SFT v5 (frozen)")
    print("Policy        : SFT v5 → DPO")

    trainer.train()

    print("=" * 70)
    print("SAVING DPO MODEL")
    print("=" * 70)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("=" * 70)
    print("DPO TRAINING COMPLETED")
    print("=" * 70)

    print(f"Output          : {OUTPUT_DIR}")
    print(
        f"Best checkpoint : "
        f"{trainer.state.best_model_checkpoint}"
    )
    print(
        f"Best eval loss  : "
        f"{trainer.state.best_metric}"
    )


if __name__ == "__main__":
    main()