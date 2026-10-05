import argparse
import json
import os

import unsloth

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import torch


MODEL_NAME = 'Qwen/Qwen3-1.7B-Base'

TRAIN_FILE = 'data/processed/sft_train.jsonl'
VALIDATION_FILE = 'data/processed/sft_validation.jsonl'

OUTPUT_DIR = 'models/sft_v10'

MAX_SEQ_LENGTH = 2048

SFT_SYSTEM_PROMPT = """You are a medical question-answering assistant.

Respond in the same language as the question.
Answer only the question asked.
Provide a clear, factual and clinically relevant answer.
Do not introduce unrelated diseases, conditions, or topics.
Do not invent medical facts, diagnoses, treatments, or recommendations.
When the available information is insufficient, state the uncertainty clearly.
Do not provide dangerous or unsupported medical recommendations."""


def parse_args():
    parser = argparse.ArgumentParser(
        description='SFT LoRA de Qwen3-1.7B-Base avec Unsloth — loss sur la completion uniquement'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Nombre d'exemples d'entraînement. Sans option = dataset complet.",
    )

    parser.add_argument(
        '--epochs',
        type=float,
        default=3.0,
        help="Nombre d'époques.",
    )

    return parser.parse_args()


def format_prompt(example):
    """
    Partie PROMPT.

    Cette partie n'est pas utilisée dans la loss avec
    completion_only_loss=True.
    """
    return (
        f'{SFT_SYSTEM_PROMPT}\n\n'
        'Question:\n'
        f"{example['instruction']}\n\n"
        'Answer:\n'
    )


def format_completion(example):
    """
    Partie COMPLETION.

    C'est cette partie qui porte la loss.
    """
    return example['response']


def load_jsonl(path):
    records = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def build_prompt_completion_dataset(records):
    """
    Transforme :

        instruction + response

    en :

        prompt + completion

    afin que TRL puisse appliquer completion_only_loss=True.
    """

    return [
        {
            'prompt': format_prompt(example),
            'completion': format_completion(example),
        }
        for example in records
    ]


def main():
    args = parse_args()

    print('=' * 70)
    print('SFT LoRA — QWEN3-1.7B-BASE + UNSLOTH')
    print('LOSS UNIQUEMENT SUR LA COMPLETION')
    print('=' * 70)

    print(f'Modèle          : {MODEL_NAME}')
    print(f'Train            : {TRAIN_FILE}')
    print(f'Validation       : {VALIDATION_FILE}')
    print(f'Sortie           : {OUTPUT_DIR}')
    print(f'Max sequence     : {MAX_SEQ_LENGTH}')
    print(f'Époques          : {args.epochs}')
    print(f'Limite train     : {args.limit}')
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible.")

    print(f'GPU              : {torch.cuda.get_device_name(0)}')
    print(f'CUDA             : {torch.version.cuda}')
    print(f'PyTorch          : {torch.__version__}')
    print(f'BF16 support     : {torch.cuda.is_bf16_supported()}')
    print()

    print('=' * 70)
    print('CHARGEMENT DES DATASETS')
    print('=' * 70)

    train_records = load_jsonl(TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)

    print(f'Train complet     : {len(train_records)}')
    print(f'Validation        : {len(validation_records)}')

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit doit être > 0')

        if args.limit > len(train_records):
            raise ValueError(
                f'--limit ({args.limit}) dépasse le nombre '
                f'd\'exemples disponibles ({len(train_records)}).'
            )

        train_records = train_records[:args.limit]

    print(f'Train utilisé     : {len(train_records)}')

    # ---------------------------------------------------------------
    # IMPORTANT :
    #
    # Ancien format :
    #
    #     {'text': 'prompt + réponse'}
    #
    # => loss sur toute la séquence.
    #
    # Nouveau format :
    #
    #     {'prompt': ..., 'completion': ...}
    #
    # + completion_only_loss=True
    #
    # => loss uniquement sur completion.
    # ---------------------------------------------------------------

    train_records = build_prompt_completion_dataset(train_records)
    validation_records = build_prompt_completion_dataset(validation_records)

    dataset = {
        'train': Dataset.from_list(train_records),
        'validation': Dataset.from_list(validation_records),
    }

    print()
    print('Exemple de prompt :')
    print('-' * 70)
    print(dataset['train'][0]['prompt'])
    print('-' * 70)

    print()
    print('Exemple de completion :')
    print('-' * 70)
    print(dataset['train'][0]['completion'])
    print('-' * 70)

    print()
    print('Format dataset :')
    print(f"  Train       : {dataset['train'].column_names}")
    print(f"  Validation  : {dataset['validation'].column_names}")

    print()
    print('=' * 70)
    print('CONFIGURATION DU MASQUAGE DE LOSS')
    print('=' * 70)

    print('Format dataset : prompt / completion')
    print('Loss prompt    : MASQUÉE')
    print('Loss completion: ACTIVE')
    print('completion_only_loss : True')
    print()

    print('=' * 70)
    print('CHARGEMENT DE QWEN3-1.7B-BASE AVEC UNSLOTH')
    print('=' * 70)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print('Modèle chargé.')
    print(f'Dtype modèle     : {next(model.parameters()).dtype}')

    print()
    print('=' * 70)
    print('CONFIGURATION LoRA')
    print('=' * 70)

    # Valeurs réellement utilisées pour l'entraînement
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.0
    lora_bias = 'none'

    lora_target_modules = [
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=lora_target_modules,
        use_gradient_checkpointing='unsloth',
        random_state=3407,
    )

    print('LoRA activé.')
    print(f'Rank (r)         : {lora_r}')
    print(f'Alpha            : {lora_alpha}')
    print(f'Dropout          : {lora_dropout}')
    print('Quantification   : NON')
    print()

    print('=' * 70)
    print('CONFIGURATION SFT')
    print('=' * 70)

    # Valeurs réellement utilisées pour l'entraînement
    learning_rate = 5e-4
    weight_decay = 0.01
    warmup_ratio = 0.05
    lr_scheduler_type = 'cosine'

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        eval_strategy='epoch',

        save_strategy='epoch',

        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        logging_steps=10,
        report_to='mlflow',

        # -----------------------------------------------------------
        # Le dataset est de type prompt/completion.
        # Loss uniquement sur completion.
        # -----------------------------------------------------------
        completion_only_loss=True,

        max_length=MAX_SEQ_LENGTH,
        packing=False,

        seed=3407,

        optim='adamw_torch_fused',

        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
    )

    print()
    print('=' * 70)
    print('DÉBUT DU SFT LoRA')
    print('=' * 70)

    print()
    print('Objectif de loss :')
    print('  Prompt     -> ignoré')
    print('  Completion -> loss active')
    print()

    train_result = trainer.train()

    best_checkpoint = trainer.state.best_model_checkpoint

    print()
    print('=' * 70)
    print('SFT TERMINÉ')
    print('=' * 70)

    print(train_result)
    print(f'Meilleur checkpoint : {best_checkpoint}')

    print()
    print('=' * 70)
    print('RÉCUPÉRATION DES MÉTRIQUES')
    print('=' * 70)

    train_metrics = dict(train_result.metrics)

    validation_history = [
        dict(entry)
        for entry in trainer.state.log_history
        if 'eval_loss' in entry
    ]

    print(f'Nombre de validations : {len(validation_history)}')

    for entry in validation_history:
        epoch = entry.get('epoch')
        eval_loss = entry.get('eval_loss')

        print(
            f'  Epoch {epoch}: '
            f'eval_loss={eval_loss}'
        )

    best_validation = None

    if validation_history:
        best_validation = min(
            validation_history,
            key=lambda x: x['eval_loss']
        )

        print()
        print('Meilleure validation :')
        print(f"  Epoch      : {best_validation.get('epoch')}")
        print(f"  Eval loss  : {best_validation.get('eval_loss')}")

    # ---------------------------------------------------------------
    # JSON FINAL
    #
    # Les valeurs de configuration correspondent maintenant
    # exactement aux valeurs réellement utilisées par le trainer.
    # ---------------------------------------------------------------

    metrics = {
        'model': MODEL_NAME,

        'training': {
            **train_metrics,
        },

        'validation': {
            'history': validation_history,
            'best': best_validation,
        },

        'model_selection': {
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'best_model_checkpoint': best_checkpoint,
        },

        'config': {
            'epochs_requested': args.epochs,
            'train_examples': len(train_records),
            'validation_examples': len(validation_records),
            'max_seq_length': MAX_SEQ_LENGTH,

            'loss': {
                'objective': 'completion_only',
                'completion_only_loss': True,
                'prompt_tokens_in_loss': False,
                'completion_tokens_in_loss': True,
            },

            'lora': {
                'r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'bias': lora_bias,

                'target_modules': lora_target_modules,
            },

            'training': {
                'per_device_train_batch_size': 1,
                'per_device_eval_batch_size': 1,
                'gradient_accumulation_steps': 8,

                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_ratio': warmup_ratio,
                'lr_scheduler_type': lr_scheduler_type,

                'bf16': True,
                'fp16': False,
                'gradient_checkpointing': True,
                'optim': 'adamw_torch_fused',
            },

            'prompt': {
                'version': 'sft_v7',
                'system_instruction': SFT_SYSTEM_PROMPT,
                'format': (
                    '{system_instruction}\n\n'
                    'Question:\n'
                    '{instruction}\n\n'
                    'Answer:\n'
                ),
            },

            'completion': {
                'format': '{response}',
            },
        },
    }

    print()
    print('=' * 70)
    print('SAUVEGARDE DU MODÈLE LoRA')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics_file = os.path.join(
        OUTPUT_DIR,
        'training_metrics.json'
    )

    with open(
        metrics_file,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f'Modèle LoRA      : {OUTPUT_DIR}')
    print(f'Métriques        : {metrics_file}')

    print()
    print('=' * 70)
    print('ENTRAÎNEMENT TERMINÉ')
    print('=' * 70)


if __name__ == '__main__':
    main()