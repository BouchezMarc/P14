from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel

import trl


# ============================================================================
# TRL 0.24.0 : STUBS OPTIONNELS
# ============================================================================

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


# ============================================================================
# CONFIGURATION EXPÉRIENCE
# ============================================================================

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

SFT_CHECKPOINT = Path(
    "models/sft_v3/checkpoint-1000"
)

TRAIN_FILE = Path(
    "data/processed/dataset_dpo_v2/anonymized/dpo_train.jsonl"
)

VALIDATION_FILE = Path(
    "data/processed/dataset_dpo_v2/anonymized/dpo_validation.jsonl"
)

OUTPUT_DIR = Path("models/dpo_v7")

MAX_LENGTH = 2048


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="DPO training from an SFT LoRA checkpoint."
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

    return parser.parse_args()


# ============================================================================
# UTILITAIRES
# ============================================================================

def path_name(path: Path) -> str:
    """
    Retourne le nom du modèle/expérience à partir du chemin.

    Exemple:
        models/sft_v3/checkpoint-1000
        -> sft_v3
    """

    return path.parent.name


def json_safe(value):
    """
    Convertit récursivement les objets courants en valeurs JSON.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ============================================================================
# CHARGEMENT JSONL
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON invalide dans {path}, "
                    f"ligne {line_no}: {exc}"
                ) from exc

            rows.append(row)

    return rows


# ============================================================================
# DATASET DPO
# ============================================================================

def build_dataset(path: Path) -> Dataset:

    rows = load_jsonl(path)

    required_fields = {
        "prompt",
        "chosen",
        "rejected",
    }

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
                f"{path}: exemple {index}: "
                "`prompt` doit être une chaîne."
            )

        if not isinstance(chosen, list):

            raise ValueError(
                f"{path}: exemple {index}: "
                "`chosen` doit être une liste de messages."
            )

        if not isinstance(rejected, list):

            raise ValueError(
                f"{path}: exemple {index}: "
                "`rejected` doit être une liste de messages."
            )

        if not prompt.strip():

            raise ValueError(
                f"{path}: exemple {index}: "
                "`prompt` est vide."
            )

        if not chosen:

            raise ValueError(
                f"{path}: exemple {index}: "
                "`chosen` est vide."
            )

        if not rejected:

            raise ValueError(
                f"{path}: exemple {index}: "
                "`rejected` est vide."
            )

        for name, messages in (
            ("chosen", chosen),
            ("rejected", rejected),
        ):

            for message_index, message in enumerate(messages):

                if not isinstance(message, dict):

                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] "
                        "doit être un objet."
                    )

                if (
                    "role" not in message
                    or "content" not in message
                ):

                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] doit contenir "
                        "`role` et `content`."
                    )

                if message["role"] not in {
                    "user",
                    "assistant",
                    "system",
                }:

                    raise ValueError(
                        f"{path}: exemple {index}: "
                        f"{name}[{message_index}] "
                        f"rôle invalide: "
                        f"{message['role']!r}"
                    )

                if not isinstance(
                    message["content"],
                    str,
                ):

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


# ============================================================================
# CHARGEMENT SFT + ADAPTER DE RÉFÉRENCE
# ============================================================================

def load_sft_model():

    if not SFT_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Checkpoint SFT introuvable:\n"
            f"  {SFT_CHECKPOINT}"
        )

    print("=" * 70)
    print("LOADING SFT MODEL")
    print("=" * 70)

    print(
        f"Base model     : {BASE_MODEL}"
    )

    print(
        f"SFT checkpoint : {SFT_CHECKPOINT}"
    )

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

    # Adapter SFT utilisé comme référence DPO.
    model.load_adapter(
        str(SFT_CHECKPOINT),
        adapter_name="reference",
    )

    if not hasattr(
        model,
        "warnings_issued",
    ):

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


# ============================================================================
# EXTRACTION DE L'HISTORIQUE
# ============================================================================

def extract_validation_history(log_history):

    validation_history = []

    for entry in log_history:

        if "eval_loss" not in entry:
            continue

        validation_history.append(
            json_safe(
                {
                    key: entry[key]
                    for key in (
                        "eval_loss",
                        "eval_runtime",
                        "eval_samples_per_second",
                        "eval_steps_per_second",
                        "epoch",
                        "step",
                    )
                    if key in entry
                }
            )
        )

    return validation_history


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    # ------------------------------------------------------------------------
    # NOMS DYNAMIQUES
    # ------------------------------------------------------------------------

    sft_name = path_name(
        SFT_CHECKPOINT
    )

    dpo_name = OUTPUT_DIR.name

    # ------------------------------------------------------------------------
    # ENVIRONNEMENT
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    print(
        f"PyTorch        : {torch.__version__}"
    )

    print(
        f"TRL            : {trl.__version__}"
    )

    print(
        f"CUDA available : {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM           : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ------------------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("LOADING DPO DATASETS")
    print("=" * 70)

    train_dataset = build_dataset(
        TRAIN_FILE
    )

    validation_dataset = build_dataset(
        VALIDATION_FILE
    )

    print(
        f"DPO train      : {len(train_dataset)}"
    )

    print(
        f"DPO validation : {len(validation_dataset)}"
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    model, tokenizer = load_sft_model()

    # ------------------------------------------------------------------------
    # CONFIGURATION DPO
    # ------------------------------------------------------------------------

    dpo_config = {

        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "beta": args.beta,
        "loss_type": "sigmoid",
        "max_length": MAX_LENGTH,
        "truncation_mode": "keep_end",
        "logging_steps": 10,
        "report_to": "mlflow",
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "adamw_torch_fused",
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "model_adapter_name": "default",
        "ref_adapter_name": "reference",
    }

    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        **dpo_config,
    )

    # ------------------------------------------------------------------------
    # TRAINER
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("STARTING DPO TRAINING")
    print("=" * 70)

    print(f"Experiment    : {dpo_name}")
    print(f"Base model    : {BASE_MODEL}")
    print(f"SFT model     : {sft_name}")
    print(f"SFT checkpoint: {SFT_CHECKPOINT}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning rate : {args.learning_rate}")
    print(f"Beta          : {args.beta}")
    print("Loss          : sigmoid DPO")
    print(f"Reference     : {sft_name} (frozen)")
    print(f"Policy        : {sft_name} → DPO")
    train_result = trainer.train()

    # ------------------------------------------------------------------------
    # SAUVEGARDE MODÈLE
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("SAVING DPO MODEL")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # ------------------------------------------------------------------------
    # MÉTRIQUES
    # ------------------------------------------------------------------------

    train_metrics = json_safe(
        dict(train_result.metrics)
    )

    validation_history = extract_validation_history(
        trainer.state.log_history
    )

    best_validation = None

    if validation_history:

        best_validation = min(
            validation_history,
            key=lambda x: x["eval_loss"],
        )

    # ------------------------------------------------------------------------
    # JSON FINAL DYNAMIQUE
    # ------------------------------------------------------------------------

    metrics = {

        "experiment": {

            "name": dpo_name,
            "base_model": BASE_MODEL,
            "sft_model": sft_name,
            "sft_checkpoint": str(
                SFT_CHECKPOINT
            ),
            "output_dir": str(
                OUTPUT_DIR
            ),
        },

        "training": train_metrics,

        "validation": {

            "history": validation_history,
            "best": best_validation,
        },

        "model_selection": {

            "load_best_model_at_end": (
                training_args.load_best_model_at_end
            ),

            "metric_for_best_model": (
                training_args.metric_for_best_model
            ),

            "greater_is_better": (
                training_args.greater_is_better
            ),

            "best_model_checkpoint": (
                trainer.state.best_model_checkpoint
            ),

            "best_metric": (
                trainer.state.best_metric
            ),
        },

        "config": {

            "dpo": {

                "beta": training_args.beta,
                "loss_type": training_args.loss_type,

                "model_adapter_name": (
                    training_args.model_adapter_name
                ),
                "ref_adapter_name": (
                    training_args.ref_adapter_name
                ),
            },

            "data": {

                "train_file": str(
                    TRAIN_FILE
                ),
                "validation_file": str(
                    VALIDATION_FILE
                ),
                "train_examples": len(
                    train_dataset
                ),
                "validation_examples": len(
                    validation_dataset
                ),
                "max_length": (
                    training_args.max_length
                ),
                "truncation_mode": (
                    training_args.truncation_mode
                ),            },

            "training": {

                "epochs_requested": (
                    training_args.num_train_epochs
                ),
                "per_device_train_batch_size": (
                    training_args.per_device_train_batch_size
                ),
                "per_device_eval_batch_size": (
                    training_args.per_device_eval_batch_size
                ),
                "gradient_accumulation_steps": (
                    training_args.gradient_accumulation_steps
                ),
                "learning_rate": (
                    training_args.learning_rate
                ),
                "weight_decay": (
                    training_args.weight_decay
                ),
                "warmup_ratio": (
                    training_args.warmup_ratio
                ),
                "lr_scheduler_type": (
                    training_args.lr_scheduler_type
                ),
                "bf16": (
                    training_args.bf16
                ),
                "fp16": (
                    training_args.fp16
                ),
                "gradient_checkpointing": (
                    training_args.gradient_checkpointing
                ),
                "optim": (
                    training_args.optim
                ),
                "seed": (
                    training_args.seed
                ),
                "data_seed": (
                    training_args.data_seed
                ),
                "logging_steps": (
                    training_args.logging_steps
                ),
                "dataloader_num_workers": (
                    training_args.dataloader_num_workers
                ),
                "remove_unused_columns": (
                    training_args.remove_unused_columns
                ),
            },
        },
    }

    # ------------------------------------------------------------------------
    # SAUVEGARDE JSON
    # ------------------------------------------------------------------------

    metrics_file = (
        OUTPUT_DIR / "training_metrics.json"
    )

    with metrics_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(metrics),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # RÉSUMÉ FINAL
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("DPO TRAINING COMPLETED")
    print("=" * 70)
    print(f"Experiment      : {dpo_name}")
    print(f"SFT model       : {sft_name}")
    print(f"Output          : {OUTPUT_DIR}")
    print(f"Metrics JSON    : {metrics_file}")
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